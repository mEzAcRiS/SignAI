"""Lectura, validacion y combinacion del dataset de landmarks.

Formato del CSV (una fila = una muestra de una mano):

    label,persona,mano,x0,y0,z0,...,x20,y20,z20
    3,Alan,Right,0.0,...

Donde las 63 columnas x0..z20 son las caracteristicas normalizadas
(origen en la muneca, escala relativa al tamano de la mano).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config


class DatasetError(ValueError):
    """Error de formato o de contenido del dataset."""


# ----------------------------------------------------------------------
# Validacion
# ----------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, *, require_label: bool = True) -> pd.DataFrame:
    """Valida la estructura del DataFrame y devuelve una copia limpia.

    Comprueba:
      * existen las 63 columnas de caracteristicas;
      * no hay valores no numericos ni NaN en las caracteristicas;
      * la columna ``label`` existe y sus valores pertenecen a CLASSES;
      * no hay filas duplicadas exactas.
    """
    if not isinstance(df, pd.DataFrame):
        raise DatasetError("Se esperaba un DataFrame de pandas.")

    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            f"Faltan {len(missing)} columnas de caracteristicas "
            f"(ejemplos: {missing[:5]})."
        )

    if require_label:
        if config.LABEL_COLUMN not in df.columns:
            raise DatasetError(f"Falta la columna '{config.LABEL_COLUMN}'.")
        labels = set(df[config.LABEL_COLUMN].astype(str).unique())
        unknown = labels - set(config.CLASSES)
        if unknown:
            raise DatasetError(
                f"Etiquetas fuera del rango permitido {config.CLASSES}: {sorted(unknown)}"
            )

    features = df[config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        n_bad = int(features.isna().any(axis=1).sum())
        raise DatasetError(f"{n_bad} fila(s) contienen valores no numericos o NaN.")

    clean = df.copy()
    clean[config.FEATURE_COLUMNS] = features.astype(np.float32)
    clean = clean.drop_duplicates().reset_index(drop=True)
    return clean


# ----------------------------------------------------------------------
# Lectura / escritura
# ----------------------------------------------------------------------
def load_dataset(path: str | Path = config.DATASET_PATH) -> pd.DataFrame:
    """Carga y valida el dataset desde un archivo CSV."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"No existe el archivo '{path}'. Ejecuta primero la recoleccion "
            f"(src/collect_data.py) o usa los datos de prueba en {config.SAMPLE_DATASET_PATH}."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # CSV corrupto o con separador invalido
        raise DatasetError(f"No se pudo leer '{path}': {exc}") from exc
    if df.empty:
        raise DatasetError(f"El archivo '{path}' esta vacio.")
    return validate_dataset(df)


def save_dataset(df: pd.DataFrame, path: str | Path = config.DATASET_PATH) -> Path:
    """Valida y guarda el dataset completo (sobrescribe)."""
    clean = validate_dataset(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(path, index=False)
    return path


def append_samples(
    samples: pd.DataFrame,
    path: str | Path = config.DATASET_PATH,
) -> pd.DataFrame:
    """Agrega muestras nuevas al CSV (crea el archivo si no existe)."""
    path = Path(path)
    if path.exists():
        existing = load_dataset(path)
        combined = pd.concat([existing, samples], ignore_index=True)
    else:
        combined = samples
    return save_dataset(combined, path)


# ----------------------------------------------------------------------
# Construccion de muestras
# ----------------------------------------------------------------------
def make_sample_row(
    features: Sequence[float],
    label: str,
    person: str = "",
    handedness: str = "Right",
) -> pd.DataFrame:
    """Crea un DataFrame de una sola fila a partir de 63 valores."""
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    if features.size != config.NUM_FEATURES:
        raise DatasetError(
            f"Se esperaban {config.NUM_FEATURES} valores, se recibieron {features.size}."
        )
    row = {
        config.LABEL_COLUMN: str(label),
        config.PERSON_COLUMN: person,
        config.HANDEDNESS_COLUMN: handedness,
    }
    row.update(dict(zip(config.FEATURE_COLUMNS, features.tolist())))
    return pd.DataFrame([row])


def merge_csvs(
    paths: Iterable[str | Path],
    output: str | Path = config.DATASET_PATH,
) -> pd.DataFrame:
    """Une varios CSV (p. ej. los de cada integrante) en un solo dataset."""
    frames = []
    for p in paths:
        frames.append(load_dataset(p))
    if not frames:
        raise DatasetError("No se indico ningun archivo para combinar.")
    merged = pd.concat(frames, ignore_index=True)
    save_dataset(merged, output)
    return validate_dataset(merged)


# ----------------------------------------------------------------------
# Utilidades para entrenamiento / pruebas
# ----------------------------------------------------------------------
def split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Separan el dataset en matriz X (63 columnas) y vector y (etiquetas)."""
    clean = validate_dataset(df)
    X = clean[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = clean[config.LABEL_COLUMN].astype(str).to_numpy()
    return X, y
