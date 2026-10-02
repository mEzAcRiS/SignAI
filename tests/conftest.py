"""Fixtures compartidas de la suite de pruebas SignAI."""

from __future__ import annotations

import numpy as np
import pytest

import config
from train import train


@pytest.fixture(scope="session")
def sample_dataset(tmp_path_factory):
    """Dataset sintetico pequeño (se genera si no existe el oficial)."""
    if config.SAMPLE_DATASET_PATH.exists():
        return config.SAMPLE_DATASET_PATH

    from make_sample_data import generate
    from dataset import save_dataset

    df = generate(per_class=60, seed=7)
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    save_dataset(df, path)
    return path


@pytest.fixture(scope="session")
def trained_bundle(sample_dataset, tmp_path_factory):
    """Modelo entrenado una sola vez para toda la sesion de pruebas."""
    out = tmp_path_factory.mktemp("models") / "model.joblib"
    return train(sample_dataset, model_choice="rf", out_path=out, verbose=False)


@pytest.fixture()
def synthetic_features() -> np.ndarray:
    """Vector de 63 valores arbitrarios (ya normalizados)."""
    rng = np.random.default_rng(0)
    features = rng.normal(0, 0.3, size=config.NUM_FEATURES).astype(np.float32)
    features[:3] = 0.0  # muneca en el origen
    return features
