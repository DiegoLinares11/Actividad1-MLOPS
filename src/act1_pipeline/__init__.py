"""act1_pipeline

Paquete de la Actividad 1: pipeline de scikit-learn para la preparación de
datos (extracción, filtrado, manejo de tipos de variables y separación del
dataset) sobre el dataset de la UEFA Champions League analizado en el
Ejercicio 1, enmarcado en las dos primeras etapas de CRISP-DM.
"""

from .data import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    PERCENT_FEATURES,
    RATIO_FEATURES,
    TARGET,
    extraer_datos,
    filtrar_datos,
    separar_datos,
)
from .pipeline import construir_pipeline, construir_preprocesamiento
from .transformers import PorcentajeATexto, RatioATexto

__all__ = [
    "NUMERIC_FEATURES",
    "PERCENT_FEATURES",
    "RATIO_FEATURES",
    "CATEGORICAL_FEATURES",
    "LEAKAGE_COLUMNS",
    "DROP_COLUMNS",
    "FEATURES",
    "TARGET",
    "extraer_datos",
    "filtrar_datos",
    "separar_datos",
    "construir_pipeline",
    "construir_preprocesamiento",
    "PorcentajeATexto",
    "RatioATexto",
]

__version__ = "0.1.0"
