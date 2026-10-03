"""Tests para el modulo de reconocimiento de movimiento (J/Z)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from motion import (
    MotionBuffer,
    MOTION_BUFFER_SIZE,
    MOTION_MIN_FRAMES,
    extract_trajectory_features,
    MotionClassifier,
    make_motion_sample_row,
    load_motion_dataset,
    split_motion_xy,
    MOTION_FEATURE_COLUMNS,
)
from hand_detector import HandDetection


def make_dummy_detection(handedness: str = "Right", tip_offset: tuple = (0.1, 0.1)) -> HandDetection:
    """Crea una deteccion falsa para testing."""
    landmarks = np.zeros((config.NUM_LANDMARKS, 3), dtype=np.float32)
    landmarks[0] = [0.5, 0.5, 0.0]  # muneca en centro
    landmarks[8] = [0.5 + tip_offset[0], 0.5 + tip_offset[1], 0.0]  # indice
    landmarks[20] = [0.5 + tip_offset[0], 0.5 + tip_offset[1], 0.0]  # menique
    
    features = np.zeros(config.NUM_FEATURES, dtype=np.float32)
    return HandDetection(
        landmarks=landmarks,
        features=features,
        handedness=handedness,
        score=0.9,
    )


def make_j_trajectory(num_frames: int = 15) -> np.ndarray:
    """Genera una trayectoria simulada de J (gancho hacia adentro con menique)."""
    t = np.linspace(0, 1, num_frames)
    # J: empieza arriba, baja, curva hacia la izquierda (gancho)
    x = 0.5 + 0.1 * np.sin(np.pi * t)  # movimiento lateral suave
    y = 0.5 - 0.2 * t + 0.05 * np.sin(2 * np.pi * t)  # baja con pequena oscilacion
    # Muneca fija
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    
    seq = np.column_stack([t, x, y, wrist_x, wrist_y]).astype(np.float32)
    return seq


def make_z_trajectory(num_frames: int = 15) -> np.ndarray:
    """Genera una trayectoria simulada de Z (zigzag horizontal con indice)."""
    t = np.linspace(0, 1, num_frames)
    # Z: tres trazos horizontales alternados
    x = 0.5 + 0.15 * np.sin(3 * np.pi * t)  # zigzag
    y = 0.5 - 0.1 * t  # baja ligeramente
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    
    seq = np.column_stack([t, x, y, wrist_x, wrist_y]).astype(np.float32)
    return seq


def make_i_trajectory(num_frames: int = 15) -> np.ndarray:
    """Genera una trayectoria simulada de I (estatico, solo menique arriba)."""
    t = np.linspace(0, 1, num_frames)
    # I: casi sin movimiento (postura estatica)
    x = 0.5 + 0.01 * np.random.randn(num_frames)
    y = 0.5 + 0.01 * np.random.randn(num_frames)
    wrist_x = np.full(num_frames, 0.5)
    wrist_y = np.full(num_frames, 0.5)
    
    seq = np.column_stack([t, x, y, wrist_x, wrist_y]).astype(np.float32)
    return seq


class TestMotionBuffer:
    """Tests para MotionBuffer."""
    
    def test_buffer_stores_detections(self):
        buffer = MotionBuffer(maxlen=5)
        det = make_dummy_detection()
        
        for i in range(3):
            buffer.add(det, "J")
        
        assert len(buffer._buffer) == 3
        assert buffer.is_ready(MOTION_MIN_FRAMES) == False  # min 10
    
    def test_buffer_maxlen(self):
        buffer = MotionBuffer(maxlen=5)
        det = make_dummy_detection()
        
        for i in range(10):
            buffer.add(det, "J")
        
        assert len(buffer._buffer) == 5  # maxlen
    
    def test_buffer_clear(self):
        buffer = MotionBuffer(maxlen=5)
        det = make_dummy_detection()
        
        buffer.add(det, "J")
        buffer.clear()
        
        assert len(buffer._buffer) == 0
        assert not buffer.is_ready(MOTION_MIN_FRAMES)
    
    def test_get_sequence(self):
        buffer = MotionBuffer(maxlen=20)
        det = make_dummy_detection()
        
        for i in range(15):
            buffer.add(det, "J")
        
        seq = buffer.get_sequence()
        assert seq is not None
        assert seq.shape == (15, 5)  # t, tip_x, tip_y, wrist_x, wrist_y
    
    def test_get_sequence_insufficient_frames(self):
        buffer = MotionBuffer(maxlen=20)
        det = make_dummy_detection()
        
        buffer.add(det, "J")
        buffer.add(det, "J")
        
        seq = buffer.get_sequence()
        assert seq is None  # menos de MOTION_MIN_FRAMES


class TestTrajectoryFeatures:
    """Tests para extraccion de features de trayectoria."""
    
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
    
    def test_extract_features_i_trajectory(self):
        seq = make_i_trajectory(15)
        features = extract_trajectory_features(seq)
        
        assert features.shape == (config.MOTION_NUM_FEATURES,)
        # Para trayectoria estatica, algunas features seran pequenas
        assert not np.any(np.isnan(features))
    
    def test_extract_features_insufficient_frames(self):
        seq = make_j_trajectory(5)  # menos de MOTION_MIN_FRAMES
        features = extract_trajectory_features(seq)
        
        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert np.all(features == 0)  # devuelve ceros
    
    def test_extract_features_none(self):
        features = extract_trajectory_features(None)
        assert features.shape == (config.MOTION_NUM_FEATURES,)
        assert np.all(features == 0)
    
    def test_features_differ_between_j_and_z(self):
        """Las features de J y Z deben ser distinguibles."""
        j_seq = make_j_trajectory(20)
        z_seq = make_z_trajectory(20)
        
        j_features = extract_trajectory_features(j_seq)
        z_features = extract_trajectory_features(z_seq)
        
        # Deben ser vectores diferentes
        diff = np.linalg.norm(j_features - z_features)
        assert diff > 1e-3  # diferencia significativa


class TestMotionClassifier:
    """Tests para MotionClassifier."""
    
    def setup_method(self):
        """Genera datos sinteticos para entrenar."""
        np.random.seed(42)
        n_samples = 100
        
        X = []
        y = []
        
        for _ in range(n_samples // 2):
            # J: trayectoria con gancho
            seq = make_j_trajectory(15 + np.random.randint(0, 5))
            features = extract_trajectory_features(seq)
            # Agregar ruido
            features += np.random.normal(0, 0.01, features.shape)
            X.append(features)
            y.append("J")
            
            # Z: trayectoria zigzag
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
        assert metrics["test_accuracy"] > 0.8  # deberia ser alto con datos sinteticos
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
        
        # Test con muestra de J
        j_seq = make_j_trajectory(20)
        j_features = extract_trajectory_features(j_seq)
        label, conf = clf.predict(j_features)
        
        assert label in ["J", "Z"]
        assert 0.0 <= conf <= 1.0
        
        # Test con muestra de Z
        z_seq = make_z_trajectory(20)
        z_features = extract_trajectory_features(z_seq)
        label, conf = clf.predict(z_features)
        
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
        
        # Cargar y verificar
        loaded = MotionClassifier.load(model_path)
        assert loaded.pipeline is not None
        assert loaded.model_type == "svm"
        assert set(loaded.classes_) == {"J", "Z"}
        
        # Verificar que predice igual
        j_seq = make_j_trajectory(20)
        j_features = extract_trajectory_features(j_seq)
        
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
        features = np.random.rand(10).astype(np.float32)  # tamaño incorrecto
        with pytest.raises(ValueError):
            make_motion_sample_row(features, "J", "Test")


# Import pd for the test above
import pandas as pd


class TestMotionIntegration:
    """Tests de integracion basicos."""
    
    def test_buffer_with_detector_output(self):
        """Verifica que MotionBuffer funciona con salida de HandDetector."""
        buffer = MotionBuffer(maxlen=20)
        
        # Simular secuencia de detecciones para J
        for i in range(15):
            det = make_dummy_detection(tip_offset=(0.1 * np.sin(i * 0.5), -0.1 * i))
            buffer.add(det, "J")
        
        assert buffer.is_ready(MOTION_MIN_FRAMES)
        seq = buffer.get_sequence()
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

    def test_train_motion_sin_datos_devuelve_error(self, monkeypatch, tmp_path, capsys):
        import train as train_mod

        monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # collected no existe
        monkeypatch.setattr(config, "MOTION_DATASET_PATH", tmp_path / "falta.csv")

        rc = train_mod.main(["--motion"])
        assert rc == 1
        assert "ERROR" in capsys.readouterr().out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])