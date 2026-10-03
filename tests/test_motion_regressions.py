"""Regresiones del flujo camara -> trayectoria -> letra mostrada, sin camara."""

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

import app
import config
import motion
from make_sample_motion_data import _GENERATORS


@pytest.mark.parametrize("letter", config.MOTION_CLASSES)
@pytest.mark.parametrize("scale", [0.08, 0.25])
def test_camera_and_synthetic_features_use_same_units(monkeypatch, letter, scale):
    seq = _GENERATORS[letter](np.random.default_rng(24))
    buffer = motion.MotionBuffer()
    clock = iter(2**20 + seq[:, 0])
    monkeypatch.setattr(motion.time, "monotonic", lambda: next(clock))
    origin = np.array([0.4, 0.6])
    for row in seq:
        lm = np.zeros((21, 3), dtype=np.float64)
        lm[0, :2] = origin + scale * row[3:5]
        lm[9, :2] = lm[0, :2] + scale * np.array([np.cos(row[5]), np.sin(row[5])])
        lm[config.MOTION_FINGER_TIP[letter], :2] = origin + scale * row[1:3]
        buffer.add(SimpleNamespace(landmarks=lm, handedness="Right"))
    captured = buffer.get_sequence(letter)
    assert captured[0, 0] == 0
    assert np.all(np.diff(captured[:, 0]) > 0)
    np.testing.assert_allclose(
        motion.extract_trajectory_features(captured),
        motion.extract_trajectory_features(seq), rtol=1e-5, atol=1e-5,
    )


def test_large_clock_origin_preserves_velocity():
    seq = _GENERATORS["J"](np.random.default_rng(0))
    later = seq.copy()
    later[:, 0] += 2**20
    np.testing.assert_allclose(
        motion.extract_trajectory_features(seq),
        motion.extract_trajectory_features(later), rtol=1e-5, atol=1e-5,
    )


def test_duplicate_timestamps_are_rejected():
    seq = _GENERATORS["J"](np.random.default_rng(0))
    seq[1, 0] = seq[0, 0]
    with pytest.raises(ValueError, match="crecientes"):
        motion.extract_trajectory_features(seq)


def _run_frames(monkeypatch, static_label, *, moving=False, model=True,
                confidence=0.95, result="J", stop=False):
    positions = [i * 0.02 if moving else 0 for i in range(20)]
    if stop:
        positions += [positions[-1]] * 25
    detections = []
    for x in positions:
        lm = np.tile([0.3 + x, 0.6, 0.0], (21, 1))
        lm[9, 1] -= 0.2
        lm[8, 1] -= 0.3
        lm[20, 1] -= 0.25
        detections.append(SimpleNamespace(landmarks=lm, features=np.zeros(63), handedness="Right"))
    stream = iter(detections)
    monkeypatch.setattr(app, "HandDetector", lambda **kw: SimpleNamespace(
        detect=lambda *a, **k: next(stream), close=lambda: None))
    calls = []

    def predict(features):
        calls.append(features)
        return result, confidence

    def load():
        if not model:
            raise FileNotFoundError("test")
        return SimpleNamespace(model_type="svm", predict=predict)

    monkeypatch.setattr(app, "load_motion_classifier", load)
    monkeypatch.setattr(app, "predict_top", lambda *a: [(static_label, 0.99)])
    monkeypatch.setattr(app, "draw_hand", lambda *a: None)
    shown = []
    monkeypatch.setattr(app, "draw_label", lambda frame, label, *a, **k: shown.append(label))
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in detections]
    stats = app.run_on_frames(frames, {}, gui=False, hold=True)
    return shown, calls, stats


@pytest.mark.parametrize("letter", config.MOTION_CLASSES)
def test_quiet_dynamic_pose_is_never_confirmed(monkeypatch, letter):
    shown, calls, stats = _run_frames(monkeypatch, letter)
    assert not calls
    assert all(not s.startswith("Sena:") for s in shown)
    assert stats["distribucion"] == {}


@pytest.mark.parametrize("letter", ["A", "I", "N", "G"])
def test_static_letters_still_work(monkeypatch, letter):
    shown, calls, stats = _run_frames(monkeypatch, letter)
    assert not calls
    assert shown[-1] == f"Sena: {letter}"


@pytest.mark.parametrize("options", [{"model": False}, {"confidence": 0.2}])
def test_unconfirmed_movement_cannot_fall_back_to_dynamic_pose(monkeypatch, options):
    shown, calls, stats = _run_frames(monkeypatch, "J", moving=True, **options)
    assert all(not s.startswith("Sena:") for s in shown)
    assert stats["con_confianza"] == 0


@pytest.mark.parametrize("base,result", [("I", "J"), ("N", "Ñ"), ("K", "K"), ("G", "Q"), ("X", "X"), ("Z", "Z")])
def test_confirmed_movement_reaches_display_and_statistics(monkeypatch, base, result):
    shown, calls, stats = _run_frames(monkeypatch, base, moving=True, result=result)
    assert calls
    assert shown[-1] == f"Sena: {result}"
    assert stats["distribucion"][result] > 0


def test_dynamic_prediction_is_cleared_after_motion_stops(monkeypatch):
    shown, calls, stats = _run_frames(monkeypatch, "J", moving=True, stop=True)
    assert "Sena: J" in shown
    assert shown[-1] == "Esperando movimiento"


def test_old_motion_data_rejected_without_overwriting(tmp_path):
    path = tmp_path / "old.csv"
    row = motion.make_motion_sample_row(np.zeros(29), "J", "person")
    row.drop(columns="feature_version").to_csv(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="antiguo"):
        motion.save_motion_samples(row, path)
    assert path.read_bytes() == original


def test_old_motion_model_rejected(tmp_path):
    path = tmp_path / "old.joblib"
    joblib.dump({"classes": ["J", "Z"]}, path)
    with pytest.raises(ValueError, match="reentrena"):
        motion.MotionClassifier.load(path)


def test_training_rejects_old_captures_instead_of_using_synthetic(monkeypatch, tmp_path, capsys):
    import train

    collected = tmp_path / "collected"
    collected.mkdir()
    row = motion.make_motion_sample_row(np.zeros(29), "J", "person")
    row.drop(columns="feature_version").to_csv(collected / "old_motion.csv", index=False)
    target = tmp_path / "merged.csv"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MOTION_DATASET_PATH", target)
    assert train.main(["--motion", "--motion-out", str(tmp_path / "model.joblib")]) == 1
    assert "antiguo" in capsys.readouterr().err
    assert not target.exists()
