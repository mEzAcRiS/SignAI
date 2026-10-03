"""Modulo de reconocimiento de movimiento para J y Z en LSM.

Este modulo implementa un clasificador de segundo nivel que analiza la
trayectoria del dedo relevante (menique para J, indice para Z) a lo
largo de una secuencia de fotogramas para distinguir J de I y Z de
otras letras con indice estirado.

Arquitectura:
    HandDetector -> 21 landmarks/frame
                    |
                    v
    MotionBuffer (ring buffer de N frames)
                    |
                    v
    extract_trajectory_features() -> vector de ~25 features
                    |
                    v
    MotionClassifier (SVM/RF) -> "J" o "Z"
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from hand_detector import HandDetection


# ----------------------------------------------------------------------
# Configuracion del buffer y features
# ----------------------------------------------------------------------
MOTION_BUFFER_SIZE = 20
MOTION_FINGER_TIP = {"J": 20, "Z": 8}  # landmark indices
MOTION_MIN_FRAMES = 10  # minimo de frames para extraer features validos


@dataclass
class MotionSample:
    """Una muestra de movimiento (secuencia de frames)."""
    features: np.ndarray      # vector de features extraidos (MOTION_NUM_FEATURES,)
    label: str               # "J" o "Z"
    person: str
    timestamp: float


class MotionBuffer:
    """Buffer circular que almacena las ultimas N detecciones de mano.
    
    Mantiene las coordenadas del landmark relevante (punta del dedo)
    y de la muneca para calcular trayectorias relativas.
    """

    def __init__(self, maxlen: int = MOTION_BUFFER_SIZE):
        self.maxlen = maxlen
        # Cada entrada: (timestamp, finger_tip_xy, wrist_xy, handedness)
        self._buffer: deque[tuple[float, np.ndarray, np.ndarray, str]] = deque(maxlen=maxlen)

    def add(self, detection: HandDetection, letter_candidate: str) -> None:
        """Agrega una deteccion al buffer.
        
        Solo guarda si la mano detectada coincide con la letra candidata
        (usamos el handedness para validar consistencia).
        """
        if detection is None:
            return
        
        finger_idx = MOTION_FINGER_TIP.get(letter_candidate)
        if finger_idx is None:
            return
        
        tip = detection.landmarks[finger_idx, :2]  # x, y
        wrist = detection.landmarks[0, :2]
        
        self._buffer.append((
            time.monotonic(),
            tip.astype(np.float32),
            wrist.astype(np.float32),
            detection.handedness
        ))

    def clear(self) -> None:
        self._buffer.clear()

    def is_ready(self, min_frames: int = MOTION_MIN_FRAMES) -> bool:
        return len(self._buffer) >= min_frames

    def get_sequence(self) -> Optional[np.ndarray]:
        """Devuelve la secuencia completa como array (N, 5): [t, tip_x, tip_y, wrist_x, wrist_y]."""
        if len(self._buffer) < MOTION_MIN_FRAMES:
            return None
        seq = np.array([
            [t, tip[0], tip[1], wrist[0], wrist[1]]
            for t, tip, wrist, _ in self._buffer
        ], dtype=np.float32)
        return seq

    def get_handedness(self) -> Optional[str]:
        if not self._buffer:
            return None
        # Usar el handedness mas frecuente en el buffer
        handednesses = [h for _, _, _, h in self._buffer]
        return max(set(handednesses), key=handednesses.count)


# ----------------------------------------------------------------------
# Extraccion de features de trayectoria
# ----------------------------------------------------------------------
def extract_trajectory_features(sequence: np.ndarray) -> np.ndarray:
    """Extrae features de una secuencia de movimiento.
    
    Entrada:
        sequence: array (N, 5) con columnas [t, tip_x, tip_y, wrist_x, wrist_y]
    
    Salida:
        vector de features (MOTION_NUM_FEATURES,)
    """
    if sequence is None or len(sequence) < MOTION_MIN_FRAMES:
        return np.zeros(config.MOTION_NUM_FEATURES, dtype=np.float32)
    
    # Coordenadas relativas a la muneca (invariante a posicion de la mano)
    tip_rel = sequence[:, 1:3] - sequence[:, 3:5]  # (N, 2)
    
    # Tiempo (normalizado a 0..1)
    t = sequence[:, 0]
    t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    
    features = []
    
    # 1. Estadisticas basicas de la trayectoria relativa
    # Centroide
    centroid = np.mean(tip_rel, axis=0)
    features.extend(centroid)  # 2
    
    # Desviacion estandar
    std_xy = np.std(tip_rel, axis=0)
    features.extend(std_xy)  # 2
    
    # Rango (bounding box)
    range_xy = np.max(tip_rel, axis=0) - np.min(tip_rel, axis=0)
    features.extend(range_xy)  # 2
    
    # 2. Velocidad
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    dxy = np.diff(tip_rel, axis=0)
    velocity = np.linalg.norm(dxy, axis=1) / dt  # velocidad escalar por frame
    features.append(float(np.mean(velocity)))   # 1
    features.append(float(np.max(velocity)))    # 1
    features.append(float(np.std(velocity)))    # 1
    
    # 3. Direccion y cambios de direccion
    # Angulo de movimiento entre frames consecutivos
    angles = np.arctan2(dxy[:, 1], dxy[:, 0])  # -pi a pi
    # Cambios de direccion (diferencia entre angulos consecutivos > 45 grados)
    angle_diffs = np.abs(np.diff(angles))
    # Normalizar a [0, pi]
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
    direction_changes = np.sum(angle_diffs > np.pi / 4)  # > 45 grados
    features.append(float(direction_changes))    # 1
    features.append(float(np.mean(angle_diffs))) # 1
    features.append(float(np.std(angle_diffs)))  # 1
    
    # 4. Geometria del trazo
    # Longitud total del trazo
    path_length = float(np.sum(np.linalg.norm(dxy, axis=1)))
    features.append(path_length)  # 1
    
    # Distancia inicio-fin
    start_end_dist = float(np.linalg.norm(tip_rel[-1] - tip_rel[0]))
    features.append(start_end_dist)  # 1
    
    # Tortuosidad (longitud / distancia inicio-fin)
    tortuosity = path_length / max(start_end_dist, 1e-6)
    features.append(tortuosity)  # 1
    
    # 5. Correlacion x-y (forma del trazo: linea vs curva)
    if len(tip_rel) > 1:
        corr = np.corrcoef(tip_rel[:, 0], tip_rel[:, 1])[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    features.append(float(corr))  # 1
    
    # 6. Momentos de la trayectoria (forma)
    # Momentos centrales normalizados
    x_centered = tip_rel[:, 0] - centroid[0]
    y_centered = tip_rel[:, 1] - centroid[1]
    
    # m20, m02, m11
    m20 = np.mean(x_centered ** 2)
    m02 = np.mean(y_centered ** 2)
    m11 = np.mean(x_centered * y_centered)
    features.extend([float(m20), float(m02), float(m11)])  # 3
    
    # Excentricidad (relacion entre autovalores de la matriz de covarianza)
    cov = np.cov(tip_rel.T)
    eigvals = np.linalg.eigvalsh(cov)  # eigvalsh para matriz simetrica = reales
    eigvals = np.sort(eigvals)[::-1]  # descendente
    if eigvals[0] > 1e-6:
        eccentricity = 1 - eigvals[1] / eigvals[0]
    else:
        eccentricity = 0.0
    features.append(float(eccentricity))  # 1
    
    # 7. Orientacion principal del trazo (angulo del primer componente principal)
    if eigvals[0] > 1e-6:
        # Eigenvector asociado al mayor autovalor (usar eigh para matriz simetrica)
        cov_norm = cov / np.trace(cov)
        eigvals_n, eigvecs = np.linalg.eigh(cov_norm)
        idx = np.argmax(eigvals_n)
        pc1 = eigvecs[:, idx]
        principal_angle = float(np.arctan2(pc1[1], pc1[0]))
    else:
        principal_angle = 0.0
    features.append(principal_angle)  # 1
    
    # 8. Duracion normalizada
    duration = float(t[-1] - t[0])
    features.append(duration)  # 1
    
    # 9. Numero de frames en la secuencia
    features.append(float(len(sequence)))  # 1
    
    # Total: 2+2+2 + 1+1+1 + 1+1+1+1 + 1 + 3+1+1 + 1 = 20 features
    # Ajustar al numero esperado
    features_arr = np.array(features, dtype=np.float32)
    
    if len(features_arr) != config.MOTION_NUM_FEATURES:
        # Pad o truncar si hay discrepancia
        if len(features_arr) < config.MOTION_NUM_FEATURES:
            features_arr = np.pad(features_arr, (0, config.MOTION_NUM_FEATURES - len(features_arr)))
        else:
            features_arr = features_arr[:config.MOTION_NUM_FEATURES]
    
    return features_arr


# ----------------------------------------------------------------------
# Clasificador de movimiento
# ----------------------------------------------------------------------
class MotionClassifier:
    """Clasificador ligero para distinguir J de Z por su trayectoria."""
    
    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        self.trained_at: Optional[str] = None
        self.metrics_: dict = {}
    
    def _build_model(self):
        if self.model_type == "svm":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=10.0, probability=True, random_state=config.RANDOM_STATE)),
            ])
        else:  # rf
            return Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                )),
            ])
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        """Entrena el clasificador de movimiento."""
        if len(np.unique(y)) < 2:
            raise ValueError("Se necesitan al menos 2 clases (J y Z) para entrenar.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
        )
        
        self.pipeline = self._build_model()
        
        # Validacion cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        
        self.classes_ = self.pipeline.classes_
        self.trained_at = time.strftime("%Y-%m-%d %H:%M:%S")
        
        test_acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(self.classes_))
        
        self.metrics_ = {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "test_accuracy": float(test_acc),
            "classification_report": report,
            "confusion_matrix": {
                "labels": sorted(self.classes_.tolist()),
                "matrix": cm.tolist(),
            },
        }
        
        if verbose:
            print(f"MotionClassifier ({self.model_type}):")
            print(f"  CV accuracy: {np.mean(cv_scores):.3f} (+-{np.std(cv_scores):.3f})")
            print(f"  Test accuracy: {test_acc:.3f}")
            print(report)
        
        return self.metrics_
    
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predice la letra (J o Z) a partir de features de trayectoria.
        
        Devuelve (etiqueta, probabilidad).
        """
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")
        
        X = np.asarray(features, dtype=np.float32).reshape(1, -1)
        proba = self.pipeline.predict_proba(X)[0]
        idx = np.argmax(proba)
        label = str(self.classes_[idx])
        confidence = float(proba[idx])
        return label, confidence
    
    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predice multiples muestras."""
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        proba = self.pipeline.predict_proba(X)
        idx = proba.argmax(axis=1)
        labels = np.array([str(c) for c in np.asarray(self.pipeline.classes_)[idx]])
        confidences = proba[np.arange(len(idx)), idx]
        return labels, confidences
    
    def save(self, path: str | Path) -> Path:
        """Guarda el modelo entrenado."""
        if self.pipeline is None:
            raise RuntimeError("No hay modelo para guardar.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": self.pipeline,
            "model_type": self.model_type,
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "trained_at": self.trained_at,
            "metrics": self.metrics_,
            "feature_names": [f"m{i}" for i in range(config.MOTION_NUM_FEATURES)],
        }
        joblib.dump(bundle, path, compress=3)
        return path
    
    @classmethod
    def load(cls, path: str | Path) -> "MotionClassifier":
        """Carga un modelo entrenado."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo de movimiento '{path}'.")
        bundle = joblib.load(path)
        clf = cls(model_type=bundle.get("model_type", "svm"))
        clf.pipeline = bundle["pipeline"]
        clf.classes_ = np.array(bundle["classes"])
        clf.trained_at = bundle.get("trained_at")
        clf.metrics_ = bundle.get("metrics", {})
        return clf


# ----------------------------------------------------------------------
# Dataset de movimiento (CSV con features de trayectoria)
# ----------------------------------------------------------------------
MOTION_FEATURE_COLUMNS = [f"m{i}" for i in range(config.MOTION_NUM_FEATURES)]
MOTION_CSV_COLUMNS = ["label", "persona"] + MOTION_FEATURE_COLUMNS


def make_motion_sample_row(
    features: np.ndarray,
    label: str,
    person: str = "",
) -> pd.DataFrame:
    """Crea un DataFrame de una fila para el CSV de movimiento."""
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    if features.size != config.MOTION_NUM_FEATURES:
        raise ValueError(f"Se esperaban {config.MOTION_NUM_FEATURES} features, se recibieron {features.size}.")
    row = {
        "label": str(label),
        "persona": person,
    }
    row.update(dict(zip(MOTION_FEATURE_COLUMNS, features.tolist())))
    return pd.DataFrame([row])


def save_motion_samples(
    samples: pd.DataFrame,
    path: str | Path = config.MOTION_DATASET_PATH,
) -> Path:
    """Guarda muestras de movimiento en CSV (agrega si existe)."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, samples], ignore_index=True)
    else:
        combined = samples
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path


def load_motion_dataset(path: str | Path = config.MOTION_DATASET_PATH) -> pd.DataFrame:
    """Carga y valida el dataset de movimiento."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset de movimiento '{path}'.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El dataset de movimiento '{path}' esta vacio.")
    
    # Validar columnas
    missing = [c for c in MOTION_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en dataset de movimiento: {missing}")
    
    # Validar labels
    labels = set(df["label"].astype(str).unique())
    unknown = labels - set(config.MOTION_CLASSES)
    if unknown:
        raise ValueError(f"Etiquetas no validas en motion dataset: {unknown}")
    
    return df


def is_motion_csv(path: str | Path) -> bool:
    """Detecta si un CSV tiene el esquema de movimiento (columnas m0..m19)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return set(MOTION_CSV_COLUMNS) <= cols


def find_motion_csvs(out_dir: str | Path) -> list[Path]:
    """Devuelve los CSV de movimiento de una carpeta (ignora los estaticos)."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    return [p for p in sorted(out_dir.glob("*.csv")) if is_motion_csv(p)]


def merge_motion_csvs(
    paths: list[str | Path],
    output: str | Path = config.MOTION_DATASET_PATH,
) -> pd.DataFrame:
    """Une varios CSV de movimiento en el dataset de movimiento final.

    Valida el esquema de cada archivo y las etiquetas (solo J/Z) antes
    de guardar; devuelve el dataset ya validado.
    """
    if not paths:
        raise ValueError("No hay CSV de movimiento para unir.")
    frames = []
    for p in paths:
        if not is_motion_csv(p):
            raise ValueError(f"'{Path(p).name}' no es un CSV de movimiento.")
        frames.append(pd.read_csv(p))
    combined = pd.concat(frames, ignore_index=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return load_motion_dataset(output)


def split_motion_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Separa el dataset de movimiento en X (features) e y (labels)."""
    X = df[MOTION_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    return X, y


# ----------------------------------------------------------------------
# Entrenamiento completo del modelo de movimiento
# ----------------------------------------------------------------------
def train_motion_model(
    data_path: str | Path = config.MOTION_DATASET_PATH,
    model_choice: str = "svm",
    out_path: str | Path = config.MOTION_MODEL_PATH,
    verbose: bool = True,
) -> dict:
    """Entrena y guarda el clasificador de movimiento J/Z.
    
    Devuelve un bundle con el modelo y metricas (compatible con load_classifier).
    """
    df = load_motion_dataset(data_path)
    X, y = split_motion_xy(df)
    
    clf = MotionClassifier(model_type=model_choice)
    clf.train(X, y, verbose=verbose)
    clf.save(out_path)
    
    # Crear bundle compatible con el formato del modelo estatico
    bundle = {
        "model": clf.pipeline,
        "model_name": f"Motion Classifier ({'SVM' if model_choice == 'svm' else 'Random Forest'})",
        "model_key": f"motion_{model_choice}",
        "classes": clf.classes_.tolist(),
        "feature_names": [f"m{i}" for i in range(config.MOTION_NUM_FEATURES)],
        "trained_at": clf.trained_at,
        "dataset_path": str(data_path),
        "n_samples": int(len(X)),
        "metrics": clf.metrics_,
    }
    
    metrics_path = Path(out_path).with_name("motion_metrics.json")
    metrics_path.write_text(
        json.dumps(clf.metrics_, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    
    if verbose:
        print(f"Modelo de movimiento guardado en: {out_path}")
        print(f"Metricas guardadas en: {metrics_path}")
    
    return bundle


def load_motion_classifier(path: str | Path = config.MOTION_MODEL_PATH) -> MotionClassifier:
    """Carga el clasificador de movimiento entrenado."""
    return MotionClassifier.load(path)