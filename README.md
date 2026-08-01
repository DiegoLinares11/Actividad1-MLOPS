# Actividad 1 · Pipeline de scikit-learn

Curso **Machine Learning Engineering (MLE/MLOps)** — Universidad del Valle de Guatemala.

Pipeline de scikit-learn para la preparación de datos, **empaquetado** para poder instalarlo y
ejecutarlo en la computadora de cualquier integrante del equipo.

> **Continuación del Ejercicio 1.** Se usa el mismo dataset (`champions_league_matches.csv`) que
> analizamos ahí, y se implementa el modelo que propusimos en ese análisis: predecir el resultado
> del partido (clasificación multiclase).

---

## Inicio rápido

```bash
git clone https://github.com/DiegoLinares11/Actividad1-MLOPS.git
cd Actividad1-MLOPS
pip install .
act1-demo
```

Salida esperada:

```
============================================================
ACTIVIDAD 1 - Pipeline de scikit-learn (UEFA Champions League)
============================================================
Paquete act1_pipeline v0.1.0
Equipo (hostname): <nombre de tu compu>
Sistema operativo: Windows 11
Python: 3.14.2

[1] Extracción      -> 151 filas, 18 columnas
[2] Filtrado        -> 144 filas, 13 columnas
[4] Separación      -> train=115  test=29

============================================================
RESULTADO - Accuracy en prueba = 0.724
============================================================
```

El `random_state` está fijo, así que **el accuracy 0.724 debe salir idéntico en todas las
computadoras**. Esa es la prueba de que el pipeline es reproducible.

> Si el comando `act1-demo` no queda en el PATH, funciona igual con:
> ```bash
> python -m act1_pipeline.cli
> ```

---

## CRISP-DM

### Business Understanding
Queremos anticipar el resultado de un partido de Champions League a partir de las estadísticas de
juego (posesión, tiros a puerta, atajadas).

- **Objetivo:** clasificar cada partido como `Home Win`, `Away Win` o `Draw`.
- **Criterio de éxito:** superar a la clase mayoritaria (el local gana el 49% de los partidos).

### Data Understanding
Del Ejercicio 1 ya sabíamos qué problemas tenía el dataset, y el pipeline los resuelve:

| Hallazgo del Ejercicio 1 | Cómo lo atiende el pipeline |
|---|---|
| 7 filas totalmente vacías que separan jornadas | Se eliminan en el filtrado (quedan **144 partidos**) |
| Posesión guardada como texto `'63%'` | Transformador `PorcentajeATexto` |
| Tiros/atajadas como texto `'3 of 10'` | Transformador `RatioATexto` (genera *hechos* e *intentos*) |
| Nulos en `home_saves_pct` (casos de "0 of 0") | Imputación por mediana |
| Equipos = variables categóricas | One-Hot Encoding |

**Fuga de información (*data leakage*):** las columnas `score` y `winner` revelan el resultado del
partido, así que se descartan. Usarlas daría un modelo con acierto casi perfecto pero inútil antes
de que el partido termine.

**Variable objetivo:** `result` — Home Win (71), Away Win (48), Draw (25).

---

## Etapas de preparación de datos

| Etapa | Dónde está | Qué hace |
|---|---|---|
| **Extracción** | `data.extraer_datos()` | Carga el CSV empaquetado con la librería. |
| **Filtrado** | `data.filtrar_datos()` + `ColumnTransformer(remainder="drop")` | Quita filas vacías, columnas con leakage y columnas irrelevantes. |
| **Manejo de tipos de variables** | `pipeline.construir_pipeline()` | Cuatro tratamientos distintos (ver abajo). |
| **Separación** | `data.separar_datos()` | `train_test_split` estratificado por `result`. |

### Manejo de los cuatro tipos de variables

| Tipo | Columnas | Tratamiento |
|---|---|---|
| Numéricas ya limpias | `*_pct` | imputar (mediana) → estandarizar |
| Porcentaje en texto | `home_possession`, `away_possession` | `'63%'` → `63.0` → imputar → estandarizar |
| Razón en texto | `*_shots_on_target`, `*_saves` | `'3 of 10'` → `3.0` y `10.0` → imputar → estandarizar |
| Categóricas | `home_team`, `away_team` | imputar (moda) → One-Hot Encoding |

Al final entran 12 columnas crudas y salen 86 columnas numéricas hacia el modelo
(`RandomForestClassifier`).

---

## Diagrama del pipeline

El diagrama se genera con `sklearn.set_config(display="diagram")` y está embebido en el notebook:

```
notebooks/pipeline_demo.ipynb
```

Para abrirlo:

```bash
pip install .[notebook]
jupyter notebook notebooks/pipeline_demo.ipynb
```

---

## Resultados

Accuracy en prueba: **0.724** (21 aciertos de 29 partidos), contra un *baseline* de 0.483 si
siempre se predijera victoria local.

| Clase | precision | recall | f1 | soporte |
|---|---|---|---|---|
| Away Win | 0.615 | 0.800 | 0.696 | 10 |
| Draw     | 0.500 | 0.200 | 0.286 | 5 |
| Home Win | 0.857 | 0.857 | 0.857 | 14 |

Las variables que más pesaron fueron los porcentajes de atajadas y de tiros a puerta.

### Limitaciones

- El conjunto de prueba tiene solo 29 partidos, así que la métrica tiene bastante margen de error.
- El modelo casi no detecta los empates (solo 1 de 5), porque hay pocos ejemplos y un empate no
  deja una huella estadística clara.
- Las variables que usa (posesión, tiros, atajadas) solo se conocen **cuando el partido ya
  terminó**, así que el modelo explica el resultado más de lo que lo predice de antemano. Para una
  predicción previa harían falta variables históricas de cada equipo.

---

## Evidencia de ejecución en distintas computadoras

La captura está al final del notebook (`notebooks/pipeline_demo.ipynb`, sección *Captura de
evidencia*). El mismo paquete se instaló y ejecutó en equipos con sistema operativo y versión de
Python diferentes, dando exactamente el mismo resultado:

| | Equipo 1 | Equipo 2 |
|---|---|---|
| Hostname | `Dlinares` | `WH-Chris` |
| Sistema operativo | Windows 11 | Linux (cachyos) |
| Python | 3.14.2 | 3.14.4 |
| **Accuracy** | **0.724** | **0.724** |

Que el resultado coincida hasta el tercer decimal, en sistemas operativos distintos, confirma que el
pipeline empaquetado es reproducible y no depende del entorno de quien lo ejecuta.

---

## Estructura del proyecto

```
.
├── pyproject.toml            # empaquetado (metadata + dependencias)
├── setup.py                  # shim de compatibilidad
├── requirements.txt
├── README.md
├── notebooks/
│   └── pipeline_demo.ipynb   # análisis + DIAGRAMA del pipeline
└── src/
    └── act1_pipeline/
        ├── __init__.py
        ├── data.py           # extracción, filtrado, separación
        ├── transformers.py   # '63%' y '3 of 10' -> números
        ├── pipeline.py       # manejo de tipos + modelo
        ├── cli.py            # ejecución de extremo a extremo
        └── datasets/
            └── champions_league_matches.csv
```

---

## Compartir el paquete

Además del `git clone`, se puede generar un instalable para pasarlo por Drive o USB:

```bash
pip install build
python -m build          # crea dist/act1_pipeline-0.1.0-py3-none-any.whl
```

El compañero lo instala con:

```bash
pip install act1_pipeline-0.1.0-py3-none-any.whl
```

El CSV va incluido dentro del paquete, así que no hace falta descargar nada aparte.

---

## Equipo

Diego Linares · Andy Fuentes · Christian Echeverria · Diederich Solis
