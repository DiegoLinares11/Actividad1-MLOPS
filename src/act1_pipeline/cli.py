"""Ejecución de extremo a extremo del pipeline.

Este script sirve como *prueba* de que el paquete se instaló y corre en una
computadora: imprime información del equipo (nombre, sistema, versión de
Python) junto con la métrica del modelo. Basta con tomarle una captura de
pantalla en cada computadora del equipo.

Uso (después de instalar el paquete):

    act1-demo
"""

from __future__ import annotations

import argparse
import datetime as _dt
import platform
import socket
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import estimator_html_repr

from . import __version__
from .data import extraer_datos, filtrar_datos, separar_datos
from .pipeline import construir_pipeline


def _banner(texto: str) -> None:
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo del pipeline (Actividad 1).")
    parser.add_argument(
        "--diagrama",
        metavar="RUTA",
        default="pipeline_diagram.html",
        help="Archivo HTML donde guardar el diagrama del pipeline.",
    )
    args = parser.parse_args(argv)

    _banner("ACTIVIDAD 1 - Pipeline de scikit-learn (UEFA Champions League)")
    print(f"Paquete act1_pipeline v{__version__}")
    print(f"Equipo (hostname): {socket.gethostname()}")
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Fecha/hora: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")

    # 1) Extracción
    df = extraer_datos()
    print(f"\n[1] Extracción      -> {df.shape[0]} filas, {df.shape[1]} columnas")

    # 2) Filtrado
    df = filtrar_datos(df)
    print(f"[2] Filtrado        -> {df.shape[0]} filas, {df.shape[1]} columnas")

    # 4) Separación
    X_train, X_test, y_train, y_test = separar_datos(df)
    print(f"[4] Separación      -> train={len(X_train)}  test={len(X_test)}")

    # 3) Manejo de tipos de variables (dentro del pipeline) + modelo
    pipe = construir_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    _banner(f"RESULTADO - Accuracy en prueba = {acc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    # Diagrama del pipeline
    ruta = Path(args.diagrama)
    ruta.write_text(estimator_html_repr(pipe), encoding="utf-8")
    print(f"Diagrama del pipeline guardado en: {ruta.resolve()}")

    print("\n[OK] Pipeline ejecutado correctamente en este equipo.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
