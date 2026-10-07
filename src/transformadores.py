"""Transformadores propios del proyecto Spotify (importados por el notebook EP2).

Se definen en un módulo, y no dentro del notebook, para que los pipelines
serializados con joblib puedan cargarse desde cualquier script.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold


def _normalizar(serie):
    return serie.astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)


class CodificadorArtista(BaseEstimator, TransformerMixin):
    """Target encoding del artista con suavizado, cross-fitting y máximo entre colaboradores.

    Entrada: DataFrame con las columnas ``artistas`` (separados por ';') y
    ``nombre_cancion`` (solo se usa para agrupar reediciones en el cross-fitting).
    Salida: una columna ``artista_popularidad``.

    - Valor de un artista = (suma_y + m * media_global) / (n_canciones + m), donde
      ``m = suavizado``. Un artista con pocas canciones se acerca a la media global.
    - Canción con varios artistas: se toma el **máximo** de sus artistas
      (una colaboración con un artista popular arrastra la popularidad).
    - Artista nunca visto: media global.
    - ``fit_transform`` (entrenamiento) usa **cross-fitting**: el valor de cada canción
      se calcula con otros pliegues, agrupados por artista + título, para que una
      canción (ni sus reediciones) nunca codifique su propia popularidad.
    - Clasificación: si ``y`` es texto, se convierte con ``mapa_clases`` a un nivel
      ordinal (bajo=0, medio=1, alto=2) y se codifica el nivel medio del artista.
    """

    def __init__(self, suavizado=10, n_pliegues=5, mapa_clases=None):
        self.suavizado = suavizado
        self.n_pliegues = n_pliegues
        self.mapa_clases = mapa_clases

    def _y_numerico(self, y):
        y = pd.Series(np.asarray(y))
        if not pd.api.types.is_numeric_dtype(y):
            if self.mapa_clases is None:
                raise ValueError('y no es numérico: indique mapa_clases.')
            y = y.map(self.mapa_clases)
        return y.astype(float).to_numpy()

    @staticmethod
    def _listas(X):
        return (X['artistas'].astype(str).str.split(';')
                .apply(lambda artistas: [a.strip().lower() for a in artistas])
                .reset_index(drop=True))

    def _tabla(self, listas, y):
        expandido = pd.DataFrame({'artista': listas.to_numpy(), 'y': y}).explode('artista')
        agregado = expandido.groupby('artista')['y'].agg(['sum', 'count'])
        media = float(np.mean(y))
        return (agregado['sum'] + self.suavizado * media) / (agregado['count'] + self.suavizado), media

    @staticmethod
    def _aplicar(listas, tabla, media):
        expandido = listas.explode()
        valores = expandido.map(tabla).astype(float).fillna(media)
        return valores.groupby(level=0).max().reindex(listas.index).to_numpy()

    def _salida(self, valores, X):
        return pd.DataFrame({'artista_popularidad': valores}, index=X.index)

    def fit(self, X, y):
        self.tabla_, self.media_ = self._tabla(self._listas(X), self._y_numerico(y))
        return self

    def transform(self, X):
        return self._salida(self._aplicar(self._listas(X), self.tabla_, self.media_), X)

    def fit_transform(self, X, y=None, **fit_params):
        y_num = self._y_numerico(y)
        listas = self._listas(X)
        grupos = (_normalizar(X['artistas']) + ' || ' + _normalizar(X['nombre_cancion'])).to_numpy()
        fuera_de_pliegue = np.empty(len(X))
        for idx_ajuste, idx_codificar in GroupKFold(n_splits=self.n_pliegues).split(listas, groups=grupos):
            tabla, media = self._tabla(listas.iloc[idx_ajuste], y_num[idx_ajuste])
            fuera_de_pliegue[idx_codificar] = self._aplicar(
                listas.iloc[idx_codificar].reset_index(drop=True), tabla, media)
        self.fit(X, y)
        return self._salida(fuera_de_pliegue, X)

    def get_feature_names_out(self, input_features=None):
        return np.array(['artista_popularidad'], dtype=object)
