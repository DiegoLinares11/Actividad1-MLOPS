"""Construcción del pipeline de scikit-learn.

El pipeline maneja los **cuatro tipos de variables** que tiene el dataset de
Champions League, cada uno con su propio tratamiento:

1. Numéricas ya limpias  -> imputar (mediana) + estandarizar.
2. Porcentajes en texto  -> convertir '63%' a 63.0 + imputar + estandarizar.
3. Razones en texto      -> convertir '3 of 10' a dos números + imputar + estandarizar.
4. Categóricas           -> imputar (moda) + One-Hot Encoding.

Al usar ``ColumnTransformer`` con ``remainder="drop"``, cualquier columna que
no esté declarada arriba se descarta automáticamente.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PERCENT_FEATURES,
    RATIO_FEATURES,
)
from .transformers import PorcentajeATexto, RatioATexto


def construir_preprocesamiento() -> ColumnTransformer:
    """Arma el preprocesamiento según el tipo de cada variable."""

    # 1) Numéricas que ya vienen como float
    pipeline_numerico = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # 2) Porcentajes guardados como texto ('63%')
    pipeline_porcentaje = Pipeline(
        steps=[
            ("parsear", PorcentajeATexto()),
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # 3) Razones guardadas como texto ('3 of 10')
    pipeline_ratio = Pipeline(
        steps=[
            ("parsear", RatioATexto()),
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # 4) Categóricas (equipos)
    pipeline_categorico = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, NUMERIC_FEATURES),
            ("porcentajes", pipeline_porcentaje, PERCENT_FEATURES),
            ("razones", pipeline_ratio, RATIO_FEATURES),
            ("categoricas", pipeline_categorico, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def construir_pipeline(modelo=None) -> Pipeline:
    """Devuelve el pipeline completo: preprocesamiento + modelo.

    El modelo por defecto es un ``RandomForestClassifier``, que funciona bien
    para la clasificación multiclase propuesta en el Ejercicio 1
    (Home Win / Away Win / Draw).
    """
    if modelo is None:
        modelo = RandomForestClassifier(n_estimators=300, random_state=42)
    return Pipeline(
        steps=[
            ("preprocesamiento", construir_preprocesamiento()),
            ("modelo", modelo),
        ]
    )
