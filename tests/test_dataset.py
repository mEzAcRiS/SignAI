"""Pruebas de carga, validacion y combinacion del dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from dataset import (
    DatasetError,
    append_samples,
    load_dataset,
    make_sample_row,
    merge_csvs,
    save_dataset,
    split_xy,
    validate_dataset,
)


def _valid_df(n: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(
            make_sample_row(
                rng.normal(0, 0.2, config.NUM_FEATURES),
                label=config.CLASSES[i % len(config.CLASSES)],
                person="test",
            ).iloc[0]
        )
    return pd.DataFrame(rows)


def test_dataset_valido():
    df = validate_dataset(_valid_df())
    assert len(df) == 10
    assert config.FEATURE_COLUMNS[0] in df.columns
    assert config.LABEL_COLUMN in df.columns


def test_falta_columna_de_caracteristicas():
    df = _valid_df().drop(columns=[config.FEATURE_COLUMNS[0]])
    with pytest.raises(DatasetError, match="Faltan"):
        validate_dataset(df)


def test_etiqueta_fuera_de_rango():
    df = _valid_df()
    df.loc[0, config.LABEL_COLUMN] = "9"
    with pytest.raises(DatasetError, match="fuera del rango"):
        validate_dataset(df)


def test_valores_no_numericos():
    df = _valid_df()
    # pandas 3.0 no deja asignar texto en una columna float: se convierte a object
    col = config.FEATURE_COLUMNS[5]
    df[col] = df[col].astype(object)
    df.loc[0, col] = "no-numerico"
    with pytest.raises(DatasetError, match="no numericos"):
        validate_dataset(df)


def test_fila_de_caracteristicas_de_tamano_erroneo():
    with pytest.raises(DatasetError, match="63"):
        make_sample_row(np.zeros(10), label=config.CLASSES[0])


def test_guardar_y_cargar(tmp_path):
    path = tmp_path / "dataset.csv"
    save_dataset(_valid_df(), path)
    df = load_dataset(path)
    assert len(df) == 10


def test_cargar_archivo_inexistente(tmp_path):
    with pytest.raises(DatasetError, match="No existe"):
        load_dataset(tmp_path / "no.csv")


def test_cargar_archivo_vacio(tmp_path):
    path = tmp_path / "vacio.csv"
    path.write_text("")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_append_no_duplica_si_se_llama_dos_veces(tmp_path):
    path = tmp_path / "d.csv"
    row = make_sample_row(np.zeros(config.NUM_FEATURES), config.CLASSES[1])
    append_samples(row, path)
    append_samples(row, path)  # fila identica -> se deduplica
    df = load_dataset(path)
    assert len(df) == 1


def test_merge_de_varios_integrantes(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    save_dataset(_valid_df(4, seed=1), a)
    save_dataset(_valid_df(6, seed=2), b)
    merged = merge_csvs([a, b], tmp_path / "merged.csv")
    assert len(merged) == 10


def test_split_xy():
    X, y = split_xy(_valid_df(12))
    assert X.shape == (12, config.NUM_FEATURES)
    assert y.shape == (12,)
    assert X.dtype == np.float32
