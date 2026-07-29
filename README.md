# Actividad 1 · Pipeline de scikit-learn

Curso **Machine Learning Engineering (MLE/MLOps)** — Universidad del Valle de Guatemala.

Este proyecto construye un **pipeline de scikit-learn** para la preparación de datos y lo
**empaqueta** para instalarlo en las computadoras del equipo.

> **Continuación del Ejercicio 1.** Se usa el mismo dataset (`champions_league_matches.csv`) que
> analizamos ahí, y se implementa el modelo que propusimos en ese análisis: predecir el resultado
> del partido (clasificación multiclase).

---

## 1. CRISP-DM (etapas de referencia)

### Business Understanding (Comprensión del negocio)
Queremos **anticipar el resultado de un partido de Champions League** a partir de las estadísticas
de juego (posesión, tiros a puerta, atajadas).

- **Objetivo:** clasificar cada partido como `Home Win`, `Away Win` o `Draw`.
- **Para qué sirve:** análisis deportivo, previas de partidos, apoyo táctico.
- **Criterio de éxito:** superar a la clase mayoritaria (el local gana ~49%) y que el pipeline sea
  reproducible por todo el equipo.

### Data Understanding (Comprensión de los datos)
Del Ejercicio 1 ya sabíamos qué problemas tenía el dataset, y el pipeline los resuelve:

| Hallazgo del Ejercicio 1 | Cómo lo atiende el pipeline |
|---|---|
| 7 filas totalmente vacías que separan jornadas | Se eliminan en el filtrado (quedan **144 partidos**) |
| Posesión guardada como texto `'63%'` | Transformador `PorcentajeATexto` |
| Tiros/atajadas como texto `'3 of 10'` | Transformador `RatioATexto` (genera *hechos* e *intentos*) |
| Nulos en `home_saves_pct` (casos de 0 de 0) | Imputación por mediana |
| Equipos = variables categóricas | One-Hot Encoding |

**Fuga de información (*data leakage*):** las columnas `score` y `winner` revelan el resultado del
partido, así que **se descartan**; usarlas daría un modelo perfecto pero inútil antes de que el
partido termine.

**Variable objetivo:** `result` — Home Win (71), Away Win (48), Draw (25).

---

## 2. Etapas de preparación de datos

| Etapa | Dónde está | Qué hace |
|---|---|---|
| **Extracción** | `data.extraer_datos()` | Carga el CSV empaquetado con la librería. |
| **Filtrado** | `data.filtrar_datos()` + `ColumnTransformer(remainder="drop")` | Quita filas vacías, columnas con leakage y columnas irrelevantes. |
| **Manejo de tipos de variables** | `pipeline.construir_pipeline()` | Cuatro tratamientos distintos (ver abajo). |
| **Separación del dataset** | `data.separar_datos()` | `train_test_split` estratificado por `result`. |

### Manejo de los cuatro tipos de variables

| Tipo | Columnas | Tratamiento |
|---|---|---|
| Numéricas ya limpias | `*_pct` | imputar (mediana) → estandarizar |
| Porcentaje en texto | `home_possession`, `away_possession` | `'63%'` → `63.0` → imputar → estandarizar |
| Razón en texto | `*_shots_on_target`, `*_saves` | `'3 of 10'` → `3.0` y `10.0` → imputar → estandarizar |
| Categóricas | `home_team`, `away_team` | imputar (moda) → One-Hot Encoding |

El **diagrama** del pipeline se genera con `sklearn.set_config(display="diagram")`; se puede ver en
`notebooks/pipeline_demo.ipynb` o abriendo `pipeline_diagram.html`.

---

## 3. Estructura del proyecto

```
actividad 1/
├── pyproject.toml            # empaquetado (metadata + dependencias)
├── setup.py                  # shim de compatibilidad
├── requirements.txt
├── README.md
├── notebooks/
│   └── pipeline_demo.ipynb   # muestra el DIAGRAMA del pipeline
└── src/
    └── act1_pipeline/
        ├── __init__.py
        ├── data.py           # extracción, filtrado, separación
        ├── transformers.py   # '63%' y '3 of 10' -> números
        ├── pipeline.py       # manejo de tipos + modelo
        ├── cli.py            # ejecución de extremo a extremo (prueba)
        └── datasets/
            └── champions_league_matches.csv
```

---

## 4. Instalación y ejecución (para cada compañero del equipo)

Dentro de la carpeta `actividad 1`:

```bash
# (opcional pero recomendado) crear un entorno virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# instalar el paquete
pip install .
```

Luego se ejecuta la demo, que imprime el **nombre del equipo, el sistema y la métrica** — sirve
como prueba de que corrió en esa computadora:

```bash
act1-demo
```

> Si el comando no aparece en el PATH, funciona igual con:
> ```bash
> python -m act1_pipeline.cli
> ```

Para ver el **diagrama**:

```bash
pip install .[notebook]
jupyter notebook notebooks/pipeline_demo.ipynb
```

---

## 5. Compartir el paquete

```bash
pip install build
python -m build          # crea dist/act1_pipeline-0.1.0-py3-none-any.whl
```

Un compañero lo instala directo con:

```bash
pip install act1_pipeline-0.1.0-py3-none-any.whl
```

El CSV va **incluido dentro del paquete**, así que no hace falta descargar nada aparte.

---

## 6. Evidencia (capturas)

Tomar **captura de pantalla de la salida de `act1-demo`** en al menos dos computadoras del equipo.
Cada captura muestra el `hostname` y la métrica, evidenciando que el mismo pipeline empaquetado
corre en diferentes máquinas.

Ejemplo de salida:

```
============================================================
ACTIVIDAD 1 - Pipeline de scikit-learn (UEFA Champions League)
============================================================
Paquete act1_pipeline v0.1.0
Equipo (hostname): Dlinares
Sistema operativo: Windows 11
Python: 3.14.2

[1] Extracción      -> 151 filas, 18 columnas
[2] Filtrado        -> 144 filas, 13 columnas
[4] Separación      -> train=115  test=29

============================================================
RESULTADO - Accuracy en prueba = 0.724
============================================================
```
