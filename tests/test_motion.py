"""Tests para el modulo de reconocimiento de movimiento (6 letras: J,K,Ñ,Q,X,Z)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from motion import (
    MotionBuffer,
    MOTION_BUFFER_SIZE,
    MOTION_MIN_FRAMES,
    MOTION_TRACE_HINTS,
    candidate_for,
    extract_trajectory_features,
    MotionClassifier,
    make_motion_sample_row,
    load_motion_dataset,
    should_use_motion,
    split_motion_xy,
    MOTION_FEATURE_COLUMNS,
)
from hand_detector import HandDetection


def make_dummy_detection(
    handedness: str = "Right",
    tip_offset: tuple = (0.1, 0.1),
    wrist: tuple = (0.5, 0.5),
    orient_deg: float = 0.0,
) -> HandDetection:
    """Crea una deteccion falsa para testing.

    Geometria: muneca -> MCP del dedo medio (9) con longitud 0.4 (escala de
    mano), rotada ``orient_deg`` respecto a la vertical; la punta del indice
    (8) y del menique (20) quedan en el MCP + ``tip_offset``.
    """
    landmarks = np.zeros((config.NUM_LANDMARKS, 3), dtype=np.float32)
    w = np.array([wrist[0], wrist[1]], dtype=np.float32)
    ang = np.deg2rad(orient_deg)
    # vector MCP - muneca (0 = dedos arriba)
    v = np.array([0.4 * np.sin(ang), -0.4 * np.cos(ang)], dtype=np.float32)
    landmarks[0, :2] = w           # muneca
    landmarks[9, :2] = w + v       # MCP del dedo medio (orientacion)
    off = np.array([tip_offset[0], tip_offset[1]], dtype=np.float32)
    landmarks[8, :2] = w + v + off    # indice
    landmarks[20, :2] = w + v + off   # menique

    features = np.zeros(config.NUM_FEATURES, dtype=np.float32)
    return HandDetection(
        landmarks=landmarks,
        features=features,
        handedness=handedness,
        score=0.9,
    )


def make_j_trajectory(num_frames: int = 15) -> np.ndarray:
    """Trayectoria simulada de J (gancho hacia adentro con menique)."""
    t = np.linspace(0, 1, num_frames)
    x = 0.5 + 0.1 * np.sin(np.pi * t)
    y = 0.5 - 0.2 * t + 0.05 * np.sin(2 * np.pi * t)
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    orient = np.full(num_frames, -np.pi / 2)
    seq = np.column_stack([t, x, y, wrist_x, wrist_y, orient]).astype(np.float32)
    return seq


def make_z_trajectory(num_frames: int = 15) -> np.ndarray:
    """Trayectoria simulada de Z (zigzag horizontal con indice)."""
    t = np.linspace(0, 1, num_frames)
    x = 0.5 + 0.15 * np.sin(3 * np.pi * t)
    y = 0.5 - 0.1 * t
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    orient = np.full(num_frames, -np.pi / 2)
    seq = np.column_stack([t, x, y, wrist_x, wrist_y, orient]).astype(np.float32)
    return seq


def make_i_trajectory(num_frames: int = 15) -> np.ndarray:
    """Trayectoria simulada de I (estatico, solo menique arriba)."""
    t = np.linspace(0, 1, num_frames)
    x = 0.5 + 0.01 * np.random.randn(num_frames)
    y = 0.5 + 0.01 * np.random.randn(num_frames)
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    orient = np.full(num_frames, -np.pi / 2) + 0.01 * np.random.randn(num_frames)
    seq = np.column_stack([t, x, y, wrist_x, wrist_y, orient]).astype(np.float32)
    return seq


def make_wrist_rocking_trajectory(num_frames: int = 15) -> np.ndarray:
    """Balanceo de muneca (tipo K): punta fija en la imagen, orienta variable."""
    t = np.linspace(0, 1, num_frames)
    tip_x = np.full(num_frames, 0.5)
    tip_y = np.full(num_frames, 0.1)
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    orient = -np.pi / 2 + 0.6 * np.sin(2 * np.pi * t)  # ±34 grados
    seq = np.column_stack([t, tip_x, tip_y, wrist_x, wrist_y, orient]).astype(np.float32)
    return seq


class TestMotionConfig:
    """Config centralizada en config.py (fuente unica)."""

    def test_six_dynamic_classes(self):
        assert set(config.MOTION_CLASSES) == {"J", "K", "Ñ", "Q", "X", "Z"}

    def test_finger_tip_for_all_classes(self):
        assert set(config.MOTION_FINGER_TIP) == set(config.MOTION_CLASSES)
        assert config.MOTION_FINGER_TIP["J"] == 20  # menique
        assert config.MOTION_FINGER_TIP["Z"] == 8   # indice

    def test_trigger_includes_twins(self):
        for letter in ("I", "N", "G"):
            assert letter in config.MOTION_TRIGGER

    def test_num_features_29(self):
        assert config.MOTION_NUM_FEATURES == 29  # 23 originales + 6 de muneca
        assert len(MOTION_FEATURE_COLUMNS) == 29

    def test_trace_hints_cover_all_classes(self):
        assert set(MOTION_TRACE_HINTS) == set(config.MOTION_CLASSES)


class TestMotionBuffer:
    """Tests para MotionBuffer."""

    def test_buffer_stores_detections(self):
        buffer = MotionBuffer(maxlen=5)
        det = make_dummy_detection()

        for _ in range(3):
            buffer.add(det)

        assert len(buffer._buffer) == 3
        assert buffer.is_ready(MOTION_MIN_FRAMES) is False  # min 10

    def test_buffer_no_letter_filter_regression_I(self):
        """Regresion: 20 detecciones etiquetadas como I deben almacenar 20
        fotogramas (antes add() descartaba por letra y el buffer quedaba en 0)."""
        buffer = MotionBuffer(maxlen=MOTION_BUFFER_SIZE)
        assert candidate_for("I") == "J"  # I dispara el analisis de movimiento

        for _ in range(20):
            buffer.add(make_dummy_detection())

        assert len(buffer._buffer) == 20  # 20 detecciones -> 20 fotogramas
        assert buffer.is_ready(MOTION_MIN_FRAMES)

    def test_buffer_maxlen(self):
        buffer = MotionBuffer(maxlen=5)
        det = make_dummy_detection()

        for _ in range(10):
            buffer.add(det)

        assert len(buffer._buffer) == 5  # maxlen

    def test_buffer_clear(self):
        buffer = MotionBuffer(maxlen=5)
        buffer.add(make_dummy_detection())
        buffer.clear()

        assert len(buffer._buffer) == 0
        assert not buffer.is_ready(MOTION_MIN_FRAMES)

    def test_get_sequence_shape(self):
        buffer = MotionBuffer(maxlen=20)
        det = make_dummy_detection()

        for _ in range(15):
            buffer.add(det)

        seq = buffer.get_sequence("J")
        assert seq is not None
        assert seq.shape == (15, 6)  # t, tip_x, tip_y, wrist_x, wrist_y, orient

    def test_get_sequence_unknown_letter(self):
        buffer = MotionBuffer(maxlen=20)
        for _ in range(15):
            buffer.add(make_dummy_detection())
        assert buffer.get_sequence("A") is None

    def test_get_sequence_insufficient_frames(self):
        buffer = MotionBuffer(maxlen=20)
        buffer.add(make_dummy_detection())
        buffer.add(make_dummy_detection())
        assert buffer.get_sequence("J") is None


class TestMovementThreshold:
    """Umbral anti-falsos-positivos (is_moving)."""

    @staticmethod
    def _fill(buffer, **kwargs):
        for _ in range(15):
            buffer.add(make_dummy_detection(**kwargs))
        return buffer

    def test_static_hand_is_not_moving(self):
        buffer = MotionBuffer(maxlen=20)
        self._fill(buffer)
        assert buffer.movement_score() < config.MOTION_MIN_MOVEMENT
        assert buffer.is_moving() is False

    def test_tiny_jitter_below_threshold(self):
        """Ruido pequeno (< umbral) NO cuenta como movimiento."""
        buffer = MotionBuffer(maxlen=20)
        for i in range(15):
            jitter = 0.01 * ((i % 3) - 1)  # ±0.01 -> rango 0.02 unidades
            buffer.add(make_dummy_detection(tip_offset=(0.1 + jitter, 0.1)))
        assert buffer.is_moving() is False

    def test_big_tip_drawing_is_moving(self):
        """Trazo de la punta (J/Z/X) supera el umbral."""
        buffer = MotionBuffer(maxlen=20)
        for i in range(15):
            dx = 0.7 * i / 14
            buffer.add(make_dummy_detection(tip_offset=(0.1 + dx, 0.1 - dx)))
        assert buffer.movement_score() >= config.MOTION_MIN_MOVEMENT
        assert buffer.is_moving() is True

    def test_wrist_rocking_is_moving(self):
        """Balanceo de muneca (K/Ñ/Q) se detecta por ORIENTACION aunque la
        punta relativa a la muneca no se mueva."""
        buffer = MotionBuffer(maxlen=20)
        for i in range(15):
            orient = -40 + 80 * i / 14  # -40 -> +40 grados
            buffer.add(make_dummy_detection(orient_deg=orient))
        assert buffer.is_moving() is True

    def test_not_ready_is_never_moving(self):
        buffer = MotionBuffer(maxlen=20)
        buffer.add(make_dummy_detection())
        assert buffer.is_moving() is False


class TestGating:
    """Regla hibrida should_use_motion / candidate_for."""

    @staticmethod
    def _moving_buffer(moving: bool = True) -> MotionBuffer:
        buffer = MotionBuffer(maxlen=20)
        for i in range(15):
            if moving:
                dx = 0.7 * i / 14
                buffer.add(make_dummy_detection(tip_offset=(0.1 + dx, 0.1)))
            else:
                buffer.add(make_dummy_detection())
        return buffer

    def test_candidate_for_twins(self):
        assert candidate_for("I") == "J"
        assert candidate_for("N") == "Ñ"
        assert candidate_for("G") == "Q"

    def test_candidate_for_dynamic_identity(self):
        for letter in config.MOTION_CLASSES:
            assert candidate_for(letter) == letter

    def test_candidate_for_other_letters(self):
        assert candidate_for("A") is None
        assert candidate_for("") is None
        assert candidate_for(None) is None

    def test_uses_motion_when_I_and_moving(self):
        assert should_use_motion("I", self._moving_buffer(moving=True)) is True

    def test_uses_motion_when_K_and_moving(self):
        assert should_use_motion("K", self._moving_buffer(moving=True)) is True

    def test_static_when_hand_quiet(self):
        """Mano quieta -> gana el clasificador estatico (anti falsos)."""
        assert should_use_motion("I", self._moving_buffer(moving=False)) is False

    def test_static_when_label_not_trigger(self):
        """Letra sin trazo (A) nunca dispara el clasificador de movimiento."""
        assert should_use_motion("A", self._moving_buffer(moving=True)) is False

    def test_static_when_buffer_not_ready(self):
        buffer = MotionBuffer(maxlen=20)
        buffer.add(make_dummy_detection())
        assert should_use_motion("I", buffer) is False


class TestTrajectoryFeatures:
    """Tests para extraccion de features de trayectoria (26)."""

    def test_extract_features_j_trajectory(self):
        seq = make_j_trajectory(15)
        features = extract_trajectory_features(seq)

        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))
        assert not np.all(features == 0)

    def test_extract_features_z_trajectory(self):
        seq = make_z_trajectory(15)
        features = extract_trajectory_features(seq)

        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))

    def test_legacy_five_column_sequence_supported(self):
        """Secuencias antiguas (N, 5) sin orientacion se aceptan."""
        legacy = make_j_trajectory(15)[:, :5]
        features = extract_trajectory_features(legacy)
        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert features[23] == 0.0  # rango de orientacion sin datos = 0

    def test_orientation_features_capture_rocking(self):
        """El giro de muneca debe verse en las features de orientacion
        (indice 23 = rango de orientacion, 27/28 = traslacion de muneca)."""
        still = extract_trajectory_features(make_i_trajectory(20))
        rock = extract_trajectory_features(make_wrist_rocking_trajectory(20))

        orient_range_still = still[23]
        orient_range_rock = rock[23]
        assert orient_range_rock > 0.5
        assert orient_range_rock > orient_range_still * 5

    def test_extract_features_insufficient_frames(self):
        seq = make_j_trajectory(5)  # menos de MOTION_MIN_FRAMES
        features = extract_trajectory_features(seq)

        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert np.all(features == 0)

    def test_extract_features_none(self):
        features = extract_trajectory_features(None)
        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert np.all(features == 0)

    def test_features_differ_between_j_and_z(self):
        """Las features de J y Z deben ser distinguibles."""
        j_features = extract_trajectory_features(make_j_trajectory(20))
        z_features = extract_trajectory_features(make_z_trajectory(20))

        diff = np.linalg.norm(j_features - z_features)
        assert diff > 1e-3


class TestMotionClassifier:
    """Tests para MotionClassifier."""

    def setup_method(self):
        """Genera datos sinteticos para entrenar."""
        np.random.seed(42)
        n_samples = 100

        X = []
        y = []

        for _ in range(n_samples // 2):
            seq = make_j_trajectory(15 + np.random.randint(0, 5))
            features = extract_trajectory_features(seq)
            features += np.random.normal(0, 0.01, features.shape)
            X.append(features)
            y.append("J")

            seq = make_z_trajectory(15 + np.random.randint(0, 5))
            features = extract_trajectory_features(seq)
            features += np.random.normal(0, 0.01, features.shape)
            X.append(features)
            y.append("Z")

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)

    def test_train_svm(self):
        clf = MotionClassifier(model_type="svm")
        metrics = clf.train(self.X, self.y, verbose=False)

        assert "test_accuracy" in metrics
        assert metrics["test_accuracy"] > 0.8
        assert clf.pipeline is not None
        assert clf.classes_ is not None
        assert set(clf.classes_) == {"J", "Z"}

    def test_train_rf(self):
        clf = MotionClassifier(model_type="rf")
        metrics = clf.train(self.X, self.y, verbose=False)

        assert "test_accuracy" in metrics
        assert metrics["test_accuracy"] > 0.8

    def test_predict(self):
        clf = MotionClassifier(model_type="svm")
        clf.train(self.X, self.y, verbose=False)

        label, conf = clf.predict(extract_trajectory_features(make_j_trajectory(20)))
        assert label in ["J", "Z"]
        assert 0.0 <= conf <= 1.0

        label, conf = clf.predict(extract_trajectory_features(make_z_trajectory(20)))
        assert label in ["J", "Z"]
        assert 0.0 <= conf <= 1.0

    def test_predict_batch(self):
        clf = MotionClassifier(model_type="svm")
        clf.train(self.X, self.y, verbose=False)

        X_test = np.vstack([
            extract_trajectory_features(make_j_trajectory(20)),
            extract_trajectory_features(make_z_trajectory(20)),
        ])

        labels, confidences = clf.predict_batch(X_test)

        assert len(labels) == 2
        assert len(confidences) == 2
        assert all(l in ["J", "Z"] for l in labels)

    def test_save_load(self, tmp_path):
        clf = MotionClassifier(model_type="svm")
        clf.train(self.X, self.y, verbose=False)

        model_path = tmp_path / "motion_model.joblib"
        clf.save(model_path)
        assert model_path.exists()

        loaded = MotionClassifier.load(model_path)
        assert loaded.pipeline is not None
        assert loaded.model_type == "svm"
        assert set(loaded.classes_) == {"J", "Z"}

        j_features = extract_trajectory_features(make_j_trajectory(20))
        label1, conf1 = clf.predict(j_features)
        label2, conf2 = loaded.predict(j_features)

        assert label1 == label2
        assert abs(conf1 - conf2) < 1e-5

    def test_insufficient_classes_raises(self):
        clf = MotionClassifier()
        X = np.random.rand(10, config.MOTION_NUM_FEATURES).astype(np.float32)
        y = np.array(["J"] * 10)  # solo una clase

        with pytest.raises(ValueError, match="al menos 2 clases"):
            clf.train(X, y)

    def test_saved_model_has_six_classes(self):
        """El modelo entrenado (si existe) debe cubrir las 6 letras."""
        if not config.MOTION_MODEL_PATH.exists():
            pytest.skip("models/motion_model.joblib no entrenado aun")
        loaded = MotionClassifier.load(config.MOTION_MODEL_PATH)
        assert set(loaded.classes_) == set(config.MOTION_CLASSES)


class TestMotionDataset:
    """Tests para funciones de dataset de movimiento."""

    def test_make_motion_sample_row(self):
        features = np.random.rand(config.MOTION_NUM_FEATURES).astype(np.float32)
        row = make_motion_sample_row(features, "J", "TestPersona")

        assert isinstance(row, pd.DataFrame)
        assert len(row) == 1
        assert row["label"].iloc[0] == "J"
        assert row["persona"].iloc[0] == "TestPersona"
        assert all(c in row.columns for c in MOTION_FEATURE_COLUMNS)

    def test_make_motion_sample_row_wrong_size(self):
        features = np.random.rand(10).astype(np.float32)
        with pytest.raises(ValueError):
            make_motion_sample_row(features, "J", "Test")

    def test_sample_motion_dataset_valid(self):
        """El dataset sintetico generado cubre las 6 letras."""
        if not config.SAMPLE_MOTION_DATASET_PATH.exists():
            pytest.skip("data/samples/motion_landmarks_sample.csv no generado")
        df = load_motion_dataset(config.SAMPLE_MOTION_DATASET_PATH)
        assert set(df["label"].astype(str)) == set(config.MOTION_CLASSES)
        X, y = split_motion_xy(df)
        assert X.shape[1] == config.MOTION_NUM_FEATURES
        assert len(np.unique(y)) == 6


class TestMotionIntegration:
    """Tests de integracion basicos."""

    def test_buffer_with_detector_output(self):
        """Verifica que MotionBuffer funciona con salida de HandDetector."""
        buffer = MotionBuffer(maxlen=20)

        for i in range(15):
            det = make_dummy_detection(tip_offset=(0.1 * np.sin(i * 0.5), -0.1 * i))
            buffer.add(det)

        assert buffer.is_ready(MOTION_MIN_FRAMES)
        seq = buffer.get_sequence("J")
        assert seq is not None
        assert seq.shape[0] == 15

        features = extract_trajectory_features(seq)
        assert features.shape == (config.MOTION_NUM_FEATURES,)


class TestMotionMerge:
    """Union de CSV estaticos y de movimiento (--merge y --motion)."""

    @staticmethod
    def _write_static_csv(path, n: int = 4):
        from dataset import make_sample_row

        rng = np.random.default_rng(0)
        rows = [
            make_sample_row(
                rng.normal(0, 0.2, config.NUM_FEATURES),
                label=config.CLASSES[i % len(config.CLASSES)],
                person="T",
            ).iloc[0]
            for i in range(n)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)

    @staticmethod
    def _write_motion_csv(path, n_per_class: int = 15):
        rows = []
        for label, base in (("J", 0.2), ("Z", 0.8)):
            for i in range(n_per_class):
                rng = np.random.default_rng(i)
                feats = (base + rng.normal(0, 0.05, config.MOTION_NUM_FEATURES)).astype(
                    np.float32
                )
                rows.append(make_motion_sample_row(feats, label, "T").iloc[0])
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_do_merge_separa_estaticos_y_movimiento(self, tmp_path, capsys):
        from collect_data import do_merge
        from dataset import load_dataset

        collected = tmp_path / "collected"
        collected.mkdir()
        self._write_static_csv(collected / "ana.csv")
        self._write_motion_csv(collected / "ana_motion.csv")
        (collected / "basura.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")

        out_static = tmp_path / "statico.csv"
        out_motion = tmp_path / "movimiento.csv"
        rc = do_merge(collected, out_static, motion_output=out_motion)
        assert rc == 0

        df = load_dataset(out_static)
        assert len(df) == 4

        mdf = load_motion_dataset(out_motion)
        assert len(mdf) == 30
        assert set(mdf["label"]) == {"J", "Z"}

        out = capsys.readouterr().out
        assert "basura.csv" in out  # ignorado con aviso de esquema desconocido

    def test_do_merge_sin_archivos(self, tmp_path, capsys):
        from collect_data import do_merge

        rc = do_merge(tmp_path / "vacio", tmp_path / "out.csv")
        assert rc == 1

    def test_train_motion_auto_merge(self, monkeypatch, tmp_path):
        """train.py --motion une los *_motion.csv si falta el dataset final."""
        import train as train_mod

        collected = tmp_path / "collected"
        collected.mkdir()
        self._write_motion_csv(collected / "ana_motion.csv")

        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "MOTION_DATASET_PATH", tmp_path / "motion_landmarks.csv")
        model_out = tmp_path / "motion_model.joblib"

        rc = train_mod.main(["--motion", "--motion-out", str(model_out)])
        assert rc == 0
        assert (tmp_path / "motion_landmarks.csv").exists()
        assert model_out.exists()

        mdf = load_motion_dataset(tmp_path / "motion_landmarks.csv")
        assert len(mdf) == 30

    def test_train_motion_falls_back_to_sample(self, monkeypatch, tmp_path, capsys):
        """Sin capturas, --motion usa los trazos sinteticos (respaldo)."""
        import train as train_mod

        monkeypatch.setattr(config, "DATA_DIR", tmp_path / "no_existe")
        monkeypatch.setattr(config, "MOTION_DATASET_PATH", tmp_path / "falta.csv")
        # el respaldo sintetico SI existe en el repo
        assert config.SAMPLE_MOTION_DATASET_PATH.exists()
        model_out = tmp_path / "motion_model.joblib"

        rc = train_mod.main(["--motion", "--motion-out", str(model_out)])
        assert rc == 0
        assert model_out.exists()
        assert "sinteticos" in capsys.readouterr().out

    def test_train_motion_sin_datos_devuelve_error(self, monkeypatch, tmp_path, capsys):
        import train as train_mod

        monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # collected no existe
        monkeypatch.setattr(config, "MOTION_DATASET_PATH", tmp_path / "falta.csv")
        monkeypatch.setattr(
            config, "SAMPLE_MOTION_DATASET_PATH", tmp_path / "sin_muestra.csv"
        )

        rc = train_mod.main(["--motion"])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
