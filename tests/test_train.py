"""Pruebas del entrenamiento, el bundle guardado y la prediccion."""

from __future__ import annotations

import json

import numpy as np
import pytest

import config
from train import load_classifier, predict_features, train


def test_train_genera_modelo_y_metricas(sample_dataset, tmp_path):
    out = tmp_path / "model.joblib"
    bundle = train(sample_dataset, model_choice="rf", out_path=out, verbose=False)

    assert out.exists()
    assert bundle["model_name"] == "Random Forest"
    assert bundle["classes"] == config.CLASSES
    assert bundle["feature_columns"] == config.FEATURE_COLUMNS
    assert bundle["n_samples"] > 0

    metrics = bundle["metrics"]
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["test_accuracy"] >= 0.8  # datos sinteticos son separables
    assert len(metrics["confusion_matrix"]["matrix"]) == len(config.CLASSES)

    # metrics.json acompana al modelo
    metrics_path = out.with_name("metrics.json")
    assert metrics_path.exists()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "all_models" in loaded and set(loaded["all_models"]) == {"rf", "svm", "mlp"}


def test_modelo_final_alternativo(trained_bundle, sample_dataset, tmp_path):
    out = tmp_path / "svm.joblib"
    bundle = train(sample_dataset, model_choice="svm", out_path=out, verbose=False)
    assert "SVM" in bundle["model_name"]


def test_load_classifier_sin_modelo(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_classifier(tmp_path / "no.joblib")


def test_predict_devuelve_clase_valida(trained_bundle, synthetic_features):
    label, confidence = predict_features(trained_bundle, synthetic_features)
    assert label in config.CLASSES
    assert 0.0 <= confidence <= 1.0


def test_predict_proba_suma(trained_bundle, synthetic_features):
    X = synthetic_features.reshape(1, -1)
    proba = trained_bundle["model"].predict_proba(X)[0]
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)


def test_prediccion_sobre_datos_del_dataset(trained_bundle, sample_dataset):
    """Predice las primeras muestras reales del CSV de prueba."""
    from dataset import load_dataset, split_xy

    X, y = split_xy(load_dataset(sample_dataset))
    correct = sum(
        predict_features(trained_bundle, row)[0] == true
        for row, true in zip(X[:50], y[:50])
    )
    assert correct >= 45  # >= 90% de aciertos
