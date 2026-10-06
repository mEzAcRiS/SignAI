"""Pruebas de la aplicacion en tiempo real (sin camara) y de predict."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from app import SignSmoother, main as app_main, run_csv
from predict import main as predict_main, predict_csv, predict_image


# ----------------------------------------------------------------------
# Suavizador de predicciones
# ----------------------------------------------------------------------
def test_smoother_rechaza_baja_confianza():
    s = SignSmoother(threshold=0.6, min_votes=2, window=5)
    assert s.update("D", 0.4) is None


def test_smoother_requiere_mayoria():
    s = SignSmoother(threshold=0.5, min_votes=3, window=5)
    s.update("A", 0.9)
    s.update("A", 0.9)
    assert s.update("A", 0.9) == "A"  # 3 votos seguidos


def test_smoother_cambia_cuando_hay_mayoria_nueva():
    s = SignSmoother(threshold=0.5, min_votes=3, window=5)
    for _ in range(3):
        s.update("A", 0.9)
    assert s.current == "A"
    for _ in range(3):
        s.update("B", 0.9)
    assert s.current == "B"


def test_smoother_olvida_si_no_hay_mano():
    s = SignSmoother(threshold=0.5, min_votes=2, window=5)
    s.update("C", 0.9)
    s.update("C", 0.9)
    assert s.current == "C"
    s.reset()
    assert s.current is None and len(s.votes) == 0


# ----------------------------------------------------------------------
# Modo CSV (sin camara) - el que usa el docente para verificar
# ----------------------------------------------------------------------
def test_run_csv(trained_bundle, sample_dataset, capsys):
    run_csv(sample_dataset, trained_bundle)
    out = capsys.readouterr().out
    assert "Exactitud sobre el CSV" in out
    assert "Distribucion suavizada" in out


def test_app_main_csv(trained_bundle, sample_dataset):
    code = app_main(["--csv", str(sample_dataset)])
    assert code == 0


def test_app_main_csv_inexistente():
    code = app_main(["--csv", "no_existe.csv"])
    assert code == 1


def test_app_main_sin_modelo(monkeypatch):
    """Si no hay modelo entrenado, la app avisa y sale con error."""
    import app

    def _raise(*args, **kwargs):
        raise FileNotFoundError(
            "No existe el modelo 'x'. Ejecuta primero: python src/train.py"
        )

    monkeypatch.setattr(app, "load_classifier", _raise)
    code = app_main(["--csv", "x.csv"])
    assert code == 1


# ----------------------------------------------------------------------
# Prediccion sobre imagen (usa data/samples/hand_test.jpg del repo)
# ----------------------------------------------------------------------
def test_predict_image_con_mano():
    image = config.SAMPLES_DIR / "hand_test.jpg"
    if not image.exists():
        pytest.skip("imagen de prueba no disponible")
    label = predict_image(image, show=False, save=False)
    assert label in config.CLASSES


def test_predict_image_inexistente():
    assert predict_image("no_existe.jpg", show=False, save=False) is None


def test_predict_csv(trained_bundle):
    """El pipeline CSV debe predecir bien sobre los datos reales del equipo."""
    if not config.DATASET_PATH.exists():
        pytest.skip("dataset real (data/hand_landmarks.csv) no disponible")
    accuracy = predict_csv(config.DATASET_PATH)
    assert accuracy >= 0.8


def test_predict_cli_requiere_fuente():
    with pytest.raises(SystemExit):
        predict_main([])
