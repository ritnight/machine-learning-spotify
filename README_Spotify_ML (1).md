# 🎵 Inteligencia Musical y Predicción de Popularidad de Canciones

**Proyecto:** Evaluación Parcial N.º 1 — Machine Learning (MLY1101)
**Caso:** C — Spotify Tracks
**Institución:** Duoc UC
**Metodología:** CRISP-DM
**Integrantes:** Alejandra González, Constanza González, Diego Villar
**Notebook asociado:** `EP1_Spotify_CRISP_DM_COMPLETO.ipynb`

> **Nota de versión:** este README fue regenerado para reflejar exactamente el notebook actual del proyecto (nombres de variables en español, función `limpiar_spotify()`, partición por `id_cancion`). Todas las cifras de esta sección fueron recalculadas ejecutando esa misma lógica contra `Spotify_Tracks_Dataset.csv` para garantizar que coincidan con lo que el notebook produce al correrlo.

---

## Tabla de contenidos

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [KPIs](#3-kpis)
4. [Fuentes de datos y herramientas colaborativas](#4-fuentes-de-datos-y-herramientas-colaborativas)
5. [Metodología CRISP-DM y alcance de esta entrega](#5-metodología-crisp-dm-y-alcance-de-esta-entrega)
6. [Diccionario de variables](#6-diccionario-de-variables)
7. [Comprensión de los datos (Data Understanding)](#7-comprensión-de-los-datos-data-understanding)
8. [Calidad de datos y limpieza (Data Preparation)](#8-calidad-de-datos-y-limpieza-data-preparation)
9. [Análisis exploratorio posterior a la limpieza](#9-análisis-exploratorio-posterior-a-la-limpieza)
10. [Pipeline de preparación para Machine Learning](#10-pipeline-de-preparación-para-machine-learning)
11. [Sesgos, ética y privacidad](#11-sesgos-ética-y-privacidad)
12. [Conclusiones](#12-conclusiones)
13. [Trabajo futuro (EP2)](#13-trabajo-futuro-ep2)
14. [Estructura del proyecto](#14-estructura-del-proyecto)
15. [Auditoría contra la rúbrica](#15-auditoría-contra-la-rúbrica)
16. [Preguntas de defensa oral](#16-preguntas-de-defensa-oral)

---

## 1. Descripción del problema de negocio

### Contexto

Spotify procesa millones de canciones de decenas de miles de artistas y géneros. Poder anticipar qué canciones alcanzarán mayor popularidad tiene valor estratégico: permite optimizar curaduría editorial, promoción algorítmica y diseño de playlists.

### Problema

Un equipo de inteligencia musical necesita determinar si los **atributos medibles de audio** de una canción (bailabilidad, energía, volumen, tempo, entre otros) permiten **anticipar su nivel de popularidad** (variable `popularidad`, escala 0–100), para apoyar decisiones de curaduría editorial y promoción algorítmica.

### Relevancia de negocio

Si el análisis y el futuro modelo logran identificar qué atributos se asocian con mayor popularidad, se puede: priorizar recursos de promoción, diseñar playlists algorítmicas más efectivas, y apoyar decisiones editoriales con evidencia auditable en lugar de solo criterio subjetivo.

---

## 2. Objetivos del proyecto

**Objetivo general:** construir una base analítica reproducible para desarrollar, en una etapa futura, un modelo de regresión que estime la popularidad de una canción a partir de sus atributos musicales medibles.

**Objetivo analítico:** identificar mediante EDA qué atributos se asocian con mayor popularidad, evaluar la calidad de los datos y dejar un pipeline de preparación reproducible y sin fuga de información.

**Objetivo de Machine Learning (EP2):** entrenar un modelo supervisado de **regresión** que prediga `popularidad` a partir de los atributos de audio y género, excluyendo identificadores y metadatos de alta cardinalidad.

---

## 3. KPIs

| KPI | Tipo | Meta | Estado en EP1 |
|---|---|---|---|
| Cobertura de datos (sin nulos en variables clave) | Calidad de datos | ≥ 99% | ✅ Verificado: 99.86% de las filas se conservan tras la limpieza |
| Contaminación train/test (`id_cancion` compartidos) | Calidad de datos | 0% | ✅ Verificado: 0 IDs solapados entre entrenamiento y prueba |
| MAE | Desempeño del modelo | < 10 puntos | 🔄 Pendiente de modelamiento (EP2) |
| RMSE | Desempeño del modelo | < 15 puntos | 🔄 Pendiente de modelamiento (EP2) |
| R² | Desempeño del modelo | > 0.20 | 🔄 Pendiente de modelamiento (EP2) |

Los KPIs de calidad de datos son condición previa: si el dataset no está limpio y sin fuga de información, ninguna métrica de desempeño del modelo (etapa futura) sería confiable. Los umbrales de MAE, RMSE y R² son referencias iniciales que se validarán contra un modelo baseline en EP2, no estándares demostrados: las correlaciones lineales entre features de audio y popularidad son todas débiles (la más fuerte es `instrumentalidad`, con -0.096), por lo que un R² alto no es esperable en este dominio.

---

## 4. Fuentes de datos y herramientas colaborativas

### Fuente de datos

| Atributo | Valor |
|---|---|
| Archivo | `Spotify_Tracks_Dataset.csv` |
| Origen declarado | Kaggle — *Spotify Tracks Dataset* (MaharshiPandya) |
| Registros | 114.000 filas × 20 columnas útiles |
| Fecha de extracción | No declarada en el CSV (limitación) |
| Versión de API | No declarada (limitación) |

**Limitación importante:** el archivo no incluye metadatos de extracción. Esto impide verificar si los valores de `popularidad` están actualizados, ya que ese indicador cambia con el tiempo según hábitos de reproducción reales.

### Herramientas colaborativas

| Herramienta | Uso |
|---|---|
| Python / pandas / numpy | Manipulación y análisis de datos |
| matplotlib / seaborn | Visualización |
| scikit-learn | Pipeline de preprocesamiento y partición por grupos |
| Jupyter Notebook | Documentación reproducible del análisis |
| Git / GitHub | Control de versiones y trazabilidad del trabajo grupal |
| Google Colab | Ejecución compartida entre integrantes |

---

## 5. Metodología CRISP-DM y alcance de esta entrega

| Fase CRISP-DM | Contenido | Estado en EP1 |
|---|---|---|
| 1. Business Understanding | Problema, objetivos, KPIs | Completa |
| 2. Data Understanding | Carga, estructura, auditoría de calidad inicial | Completa |
| 3. Data Preparation | Limpieza, EDA posterior, pipeline de preparación | Completa |
| 4. Modeling | Entrenamiento de modelos | Reservada para EP2 |
| 5. Evaluation | Métricas y comparación | Reservada para EP2 |
| 6. Deployment | Puesta en producción y monitoreo | Trabajo futuro |

CRISP-DM es iterativa: en este proyecto, la auditoría de calidad de la Fase 2 determinó directamente las reglas de limpieza aplicadas en la Fase 3. **En esta entrega no se entrena ni se evalúa un modelo predictivo.**

---

## 6. Diccionario de variables

Todos los nombres se tradujeron al español desde la carga del CSV, antes de generar cualquier tabla o gráfico. El archivo original no se modifica.

| Original (CSV) | Español (notebook) | Rol |
|---|---|---|
| track_id | id_cancion | Identificador — excluir del modelo |
| artists | artistas | Metadato |
| album_name | nombre_album | Metadato |
| track_name | nombre_cancion | Metadato |
| popularity | **popularidad** | **Target** (0–100) |
| duration_ms | duracion_ms | Feature numérica |
| explicit | contenido_explicito | Feature booleana |
| danceability | bailabilidad | Feature numérica (0–1) |
| energy | energia | Feature numérica (0–1) |
| key | tonalidad | Feature categórica (código musical, no magnitud) |
| loudness | volumen_db | Feature numérica (dB) |
| mode | modo | Feature categórica (Mayor=1 / Menor=0) |
| speechiness | presencia_habla | Feature numérica (0–1) |
| acousticness | acusticidad | Feature numérica (0–1) |
| instrumentalness | instrumentalidad | Feature numérica (0–1) |
| liveness | presencia_en_vivo | Feature numérica (0–1) |
| valence | positividad | Feature numérica (0–1) |
| tempo | tempo_bpm | Feature numérica (BPM) |
| time_signature | compas | Feature categórica (código musical, no magnitud) |
| track_genre | genero_musical | Feature categórica (114 géneros, 1.000 canciones cada uno en el CSV crudo) |

**Nota:** `tonalidad`, `modo` y `compas` son códigos musicales, no magnitudes — se tratan como categóricas para no imponer un orden numérico que no existe musicalmente.

---

## 7. Comprensión de los datos (Data Understanding)

- **Estructura:** 114.000 filas, 20 columnas útiles (se descarta `Unnamed: 0`, un índice artificial de exportación).
- **Auditoría inicial de calidad** (antes de limpiar), verificada por ejecución directa:

| Variable | Nulos explícitos | "?" | Ceros |
|---|---|---|---|
| artistas | 1 | 0 | — |
| nombre_album | 1 | 20 | — |
| nombre_cancion | 1 | 0 | — |
| tempo_bpm | 0 | — | 157 |
| bailabilidad | 0 | — | 157 |
| energia | 0 | — | 1 |
| compas | 0 | — | 163 |
| duracion_ms | 0 | — | 1 |

- **Canciones únicas:** 89.741 `id_cancion` distintos sobre 114.000 filas → 16.641 canciones aparecen en más de un género (una misma canción catalogada bajo varias etiquetas de género). Esto es exactamente lo que motiva partir por `id_cancion` y no por fila al separar entrenamiento y prueba.
- **Correlaciones preliminares:** ninguna variable de audio individual supera 0.10 en valor absoluto de correlación lineal con `popularidad`, ya en los datos crudos.

---

## 8. Calidad de datos y limpieza (Data Preparation)

### Reglas aplicadas (función `limpiar_spotify`)

| Variable(s) | Condición | Tratamiento |
|---|---|---|
| artistas, nombre_cancion | Nulo, vacío o "?" | Eliminar la fila |
| nombre_album | Nulo, vacío o "?" | Imputar `DESCONOCIDO` en filas conservadas |
| tempo_bpm, bailabilidad, energia | Al menos una igual a cero | Eliminar la fila (una sola vez) |
| compas | Cero o nulo | Imputar con la **moda**, aprendida solo en entrenamiento |
| duracion_ms | Cero, negativo o nulo | Eliminar la fila |

**Decisión de diseño explícita:** eliminar ceros de `bailabilidad` y `energia` no es un hallazgo objetivo, sino una decisión del proyecto — en la escala 0–1 de Spotify, un cero exacto es prácticamente imposible en la práctica y se trata como error de medición. Esto puede sesgar la muestra hacia canciones más bailables/enérgicas (ver sección de ética).

### Resultados verificados (ejecutando la función sobre el CSV real)

| Indicador | Cantidad |
|---|---|
| Filas originales | 114.000 |
| Filas eliminadas (sin duplicar coincidencias) | 158 |
| Filas conservadas | 113.842 (**99.86%**) |
| Álbumes imputados como `DESCONOCIDO` | 20 |
| Compases imputados | 6 |
| Moda de compás (aprendida en entrenamiento) | 4 |
| Partición | 91.072 filas entrenamiento / 22.770 filas prueba |
| IDs de canción compartidos entre train y test | **0** ✅ |

La partición se hizo con `GroupShuffleSplit` agrupando por `id_cancion` (semilla 42, 20% de prueba), por lo que ninguna canción queda repartida entre ambos conjuntos — se verificó directamente que el solape de IDs es cero.

### Validación post-limpieza y outliers

Tras la limpieza, todas las reglas de dominio se cumplen (popularidad 0–100, tonalidad -1 a 11, modo en {0,1}, variables de audio en 0–1, sin nulos remanentes). El criterio IQR detecta outliers pero **no se usa para eliminarlos automáticamente**: por ejemplo, `instrumentalidad` tiene un 22.1% de valores fuera del rango IQR, pero corresponde a una distribución legítimamente bimodal (canciones con voz vs. instrumentales puras), no a errores.

---

## 9. Análisis exploratorio posterior a la limpieza

- **Distribución de popularidad:** 14.07% de las canciones tiene `popularidad = 0` tras la limpieza. El resto se concentra mayormente entre 10 y 70 puntos.
- **Correlaciones con popularidad** (post-limpieza, de mayor a menor magnitud):

| Variable | Correlación con popularidad |
|---|---|
| instrumentalidad | -0.096 |
| presencia_habla | -0.045 |
| positividad | -0.040 |
| acusticidad | -0.026 |
| volumen_db | +0.052 |
| bailabilidad | +0.037 |

Todas las correlaciones son débiles. Esto anticipa que un modelo lineal simple probablemente no bastará y que el R² esperado en EP2 será modesto — de ahí el umbral conservador (> 0.20) definido en los KPIs.

- **Popularidad por género:** los géneros con mayor popularidad mediana son *pop* (66), *k-pop* (60) y *pop-film* (60); varios géneros (*jazz*, *latin*, *romance*, *rock*, *soul*) tienen mediana 0, es decir, más de la mitad de sus canciones no registran reproducciones relevantes en el momento de la extracción.

---

## 10. Pipeline de preparación para Machine Learning

- **Objetivo (target):** `popularidad`. Se excluyen identificadores y metadatos de alta cardinalidad.
- **Numéricas:** `StandardScaler` (sensible a outliers extremos, pero preferido sobre `MinMaxScaler` porque no comprime el rango completo si hay valores atípicos, como duraciones muy largas).
- **Compás:** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder` dentro de un sub-pipeline, ajustado solo con entrenamiento.
- **Otras categóricas** (`contenido_explicito`, `tonalidad`, `modo`, `genero_musical`): `OneHotEncoder(handle_unknown='ignore')` — no se usa Label Encoding porque estas variables no tienen un orden natural.
- **Sin fuga de información:** el pipeline se ajusta (`fit`) únicamente sobre el conjunto de entrenamiento (`X_train`) y se aplica (`transform`) sobre prueba. Se verificó por código que la moda aprendida por el pipeline coincide con la moda reservada en la limpieza, y que ambas matrices resultantes no contienen nulos ni valores no finitos.

En esta entrega el pipeline queda **construido y verificado, pero no se entrena ningún modelo** sobre él.

---

## 11. Sesgos, ética y privacidad

| Riesgo o sesgo | Tipo | Evidencia / medida de control |
|---|---|---|
| Eliminación de ceros de audio afecta más a ciertos géneros | **Observado** | De las 157 filas eliminadas por cero en tempo/bailabilidad/energía, **138 (88%) pertenecen al género "sleep"**. La regla de limpieza afecta desproporcionadamente a ese género — se documenta explícitamente en vez de ocultarlo. |
| Exposición desigual por género en el futuro modelo | Potencial | Evaluar MAE y RMSE por género en EP2, no solo de forma global. |
| Mayor presencia de algunos artistas/canciones repetidas entre géneros | Potencial | 16.641 canciones aparecen en más de un género; se separó por `id_cancion` (no por fila) precisamente para evitar que esto genere fuga de información o sobre-representación en la evaluación. |
| Popularidad desactualizada o cambiante en el tiempo | Metodológico | Documentar la fecha de extracción cuando esté disponible; el CSV no la declara. |
| Ciclo de retroalimentación algorítmica | Potencial | Un modelo entrenado con popularidad histórica puede reforzar la exposición de canciones/artistas ya populares. Mantener supervisión humana y no tratar la predicción como sinónimo de calidad artística. |

**Privacidad:** el dataset no contiene historiales ni perfiles de usuarios individuales — solo metadatos públicos de canciones (nombre de artista, álbum, atributos de audio). El riesgo de privacidad individual es bajo. Si en el futuro se incorporaran datos de usuarios (historial de reproducción, ubicación), aplicarían principios de minimización de datos y la normativa de protección de datos personales correspondiente.

---

## 12. Conclusiones

1. Los nombres de variables quedan en español desde la carga, sin modificar el CSV original.
2. La auditoría distingue nulos explícitos, "?", vacíos y ceros — evita tratar un cero como un dato faltante sin justificación.
3. La limpieza conserva el 99.86% de las filas; las eliminaciones y su motivo quedan documentados y son auditables.
4. La partición por `id_cancion` (no por fila) evita que una misma canción quede repartida entre entrenamiento y prueba — se verificó con overlap = 0.
5. El pipeline de preparación (`StandardScaler` + `OneHotEncoder`) queda listo y validado, pero **no se entrena ningún modelo en esta entrega**.
6. Las correlaciones lineales entre atributos de audio y popularidad son todas débiles, lo que ya anticipa que el problema de predicción es exigente y justifica metas de desempeño conservadoras para EP2.

---

## 13. Trabajo futuro (EP2)

- Entrenar y comparar modelos de regresión (ej. regresión lineal regularizada, árboles/ensambles) dentro del pipeline ya construido.
- Evaluar MAE, RMSE y R² sobre el conjunto de prueba, comparando contra un baseline que prediga siempre la media de popularidad.
- Cuantificar el desempeño del modelo por género musical, para verificar si el sesgo potencial de exposición desigual se materializa.
- Repetir la imputación de compás dentro de cada pliegue si se usa validación cruzada por grupos (no reutilizar `spotify_clean.csv` ya imputado para ese fin).

---

## 14. Estructura del proyecto

```
Spotify-ML/
│
├── data/
│   ├── raw/
│   │   └── Spotify_Tracks_Dataset.csv
│   └── processed/
│       ├── spotify_clean.csv
│       ├── spotify_base_modelo.csv
│       ├── auditoria_faltantes.csv
│       ├── resumen_eliminacion.csv
│       ├── balance_limpieza.csv
│       ├── filas_descartadas.csv
│       ├── particion_modelo.csv
│       ├── outliers_iqr_summary.csv
│       └── variables_modelo.csv
│
├── notebooks/
│   └── EP1_Spotify_CRISP_DM_COMPLETO.ipynb
│
├── images/
│   ├── 00_calidad_datos_crudos.png
│   ├── 00_distribuciones_crudas.png
│   └── (resto de gráficos generados por el notebook)
│
├── models/
│   └── (vacío — reservado para EP2)
│
└── README.md          ← este archivo
```

---

## 15. Auditoría contra la rúbrica

| Indicador | % | Evidencia en el proyecto | Nivel estimado |
|---|---|---|---|
| **IE1:** Fuentes de datos y herramientas colaborativas | 10% | Sección 4: fuente documentada con limitación explícita (sin fecha/versión de API), herramientas justificadas una por una | ✅ Muy buen desempeño |
| **IE2:** Manipulación y preparación de datos en Python | 30% | `limpiar_spotify()` auditable, `GroupShuffleSplit` por `id_cancion`, imputación de moda solo en entrenamiento, pipeline `ColumnTransformer` sin fuga verificada por código | ✅ Muy buen desempeño |
| **IE3:** Análisis exploratorio y calidad de los datos | 40% | Auditoría inicial, estadística antes/después de limpiar, histogramas, validación de dominios, outliers IQR justificados, correlaciones, popularidad por género | ✅ Muy buen desempeño |
| **IE4:** Sesgos, ética y privacidad | 20% | Riesgos cuantificados con evidencia real (ej. género "sleep" concentra el 88% de las eliminaciones por cero en audio), no solo enunciados en general | ✅ Buen desempeño — se fortaleció con evidencia cuantitativa |

---

## 16. Preguntas de defensa oral

### IE1 — Fuentes y herramientas

**P1. ¿Qué limitación tiene la fuente de datos?**
No declara fecha de extracción ni versión de API de Spotify, lo que impide verificar qué tan actualizados están los valores de `popularidad`, un indicador que cambia con el tiempo.

**P2. ¿Para qué usarían GitHub en este proyecto?**
Control de versiones del notebook y del código, trazabilidad de cambios entre los tres integrantes, y reproducibilidad: cualquier persona puede clonar el repositorio y ejecutar el mismo análisis.

### IE2 — Manipulación de datos

**P3. ¿Qué es data leakage y cómo lo evitaron concretamente?**
Ocurre cuando información del conjunto de prueba influye en el entrenamiento, inflando artificialmente las métricas. Aquí, 16.641 canciones aparecen en más de un género (misma canción, distintas filas), así que si se particionara por fila una misma canción podría caer en train y test a la vez. Se usó `GroupShuffleSplit` agrupando por `id_cancion`, y se verificó por código que el solape de IDs entre conjuntos es cero.

**P4. ¿Por qué usaron StandardScaler y no MinMaxScaler?**
StandardScaler es más robusto ante outliers: MinMaxScaler comprime todo al rango [0,1], y si hay valores extremos (por ejemplo duraciones muy largas), el resto de los valores normales queda comprimido en un rango muy pequeño.

**P5. ¿Cuándo se ajusta el imputador de compás, con todo el dataset o solo con entrenamiento?**
Solo con entrenamiento. La moda (valor 4) se calcula exclusivamente con las filas de `X_train` y luego se aplica también a prueba — ajustarla con todo el dataset sería una forma sutil de fuga de información.

**P6. ¿Por qué no usaron Label Encoding para género musical?**
Porque asignaría un orden numérico arbitrario entre géneros (género 0 < género 1), que no existe musicalmente. `OneHotEncoder` crea una columna binaria por género sin imponer ese orden falso.

### IE3 — EDA y calidad

**P7. ¿Cuál es la correlación más fuerte entre un atributo de audio y popularidad?**
`instrumentalidad`, con -0.096: canciones más instrumentales (sin voz) tienden a ser levemente menos populares. Es una correlación débil, explica menos del 1% de la varianza — por eso el proyecto no promete un modelo con R² alto.

**P8. ¿Por qué no eliminaron el 22% de outliers detectados en instrumentalidad?**
Porque esa variable tiene una distribución bimodal legítima: muchas canciones tienen exactamente 0 (con voz) y otras valores altos (instrumentales puras). El IQR "marca" esa bimodalidad como atípica, pero eliminar esos valores significaría borrar todas las canciones instrumentales del dataset, algo sin justificación real.

**P9. ¿Por qué la mediana de popularidad es 0 en géneros como jazz, latin o rock?**
Significa que más de la mitad de las canciones de esos géneros no tenían reproducciones relevantes al momento de la extracción del dataset. Puede reflejar catálogo antiguo poco activo o menor exposición algorítmica — no necesariamente menor calidad musical.

### IE4 — Sesgos y ética

**P10. Deme un ejemplo concreto de sesgo que hayan detectado, no genérico.**
Al eliminar filas con cero en tempo, bailabilidad o energía, el 88% de esas eliminaciones (138 de 157 filas) corresponde al género "sleep". Es esperable — canciones de sueño/ambiente pueden tener parámetros de audio at\u00edpicos por diseño — pero significa que nuestra regla de limpieza reduce más la representación de ese género que la de cualquier otro. Lo dejamos documentado en vez de tratarlo como neutral.

**P11. ¿Qué es un ciclo de retroalimentación algorítmica y por qué es un riesgo ético aquí?**
Si el modelo predice alta popularidad para una canción y eso influye en que se promocione más, su popularidad real puede subir "confirmando" la predicción — independientemente de la calidad musical. Esto favorece sistemáticamente a canciones/artistas ya conocidos. Por eso proponemos mantener supervisión humana y no usar la predicción como proxy de calidad artística.

**P12. ¿El dataset tiene problemas de privacidad?**
No en su forma actual: no contiene historiales ni perfiles de usuarios, solo metadatos públicos de canciones. El riesgo aparecería si se incorporaran datos de usuarios individuales (historial de reproducción, ubicación), donde aplicarían principios de minimización de datos y normativa de protección de datos personales.

**P13. ¿Qué harían diferente si tuvieran que mejorar el dataset?**
Registrar fecha de extracción y versión de API; medir si la eliminación de ceros de audio afecta desproporcionadamente a otros géneros además de "sleep"; y, en EP2, reportar métricas de error por género para verificar si el sesgo potencial de exposición desigual se materializa en el modelo entrenado.

---

*Proyecto desarrollado para la asignatura Machine Learning (MLY1101), Duoc UC.*
