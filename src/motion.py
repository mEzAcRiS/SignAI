"""Modulo de reconocimiento de movimiento para las letras dinamicas del LSM.

El alfabeto LSM tiene 6 letras dinamicas: J, K, Ñ, Q, X, Z. Este modulo
implementa un clasificador de segundo nivel que analiza la trayectoria de
la mano a lo largo de ~20 fotogramas para decidir cual de las 6 es, solo
cuando hay movimiento real (umbral anti-falsos-positivos).

Arquitectura:
    HandDetector -> 21 landmarks/frame
                    |
                    v
    MotionBuffer (ring buffer de N frames, guarda los 21 landmarks)
        |  is_moving()  -> hay movimiento real? (punta + orientacion + muneca)
        |  get_sequence(letra) -> [t, tip, wrist, orient] (N, 6)
                    |
                    v
    extract_trajectory_features() -> vector de 29 features
                    |
                    v
    MotionClassifier (SVM/RF) -> "J" | "K" | "Ñ" | "Q" | "X" | "Z"

Nota: K, Ñ y Q se mueven con la MUNECA (balanceo/pivote), por eso las
features incluyen orientacion de la mano y trayectoria de la muneca, no
solo la punta del dedo.
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
# Configuracion del buffer y features (unica fuente: config.py)
# ----------------------------------------------------------------------
MOTION_BUFFER_SIZE = config.MOTION_BUFFER_SIZE
MOTION_MIN_FRAMES = config.MOTION_MIN_FRAMES
MOTION_FINGER_TIP = config.MOTION_FINGER_TIP

# Puntas de los cuatro dedos largos: con ellas se mide si hay movimiento
# (cualquier dedo en movimiento cuenta como "mano en movimiento").
_ALL_FINGER_TIPS = (8, 12, 16, 20)
# Landmark 0 = muneca, 9 = MCP del dedo medio (define la orientacion).
_WRIST, _MCP_MIDDLE = 0, 9

# Trazos de cada letra dinamica (para la pantalla de captura y los docs).
MOTION_TRACE_HINTS: dict[str, str] = {
    "J": "Manique arriba, dibuja J: baja y enganche a la izquierda",
    "K": "Config K (V) y balancea la muneca arriba-abajo",
    "Ñ": "Config N y balancea la muneca lado a lado (virgulilla)",
    "Q": "Gatillo abajo y pivotea la muneca izquierda-derecha",
    "X": "Gancho del indice y rasgueo hacia ti (traccion)",
    "Z": "Indice: horizontal, diagonal abajo, horizontal",
}


@dataclass
class MotionSample:
    """Una muestra de movimiento (secuencia de frames)."""
    features: np.ndarray      # vector de features extraidos (MOTION_NUM_FEATURES,)
    label: str               # letra dinamica: J, K, Ñ, Q, X o Z
    person: str
    timestamp: float


def normalize_motion_sequence(sequence: np.ndarray, hand_scale: float = 1.0) -> np.ndarray:
    """Tiempo relativo y coordenadas en unidades de mano, en float64.

    hand_scale es la distancia media muneca-MCP medio en la secuencia.
    Los generadores sinteticos ya usan estas unidades (escala 1).
    Se conserva la traslacion de la muneca respecto al primer fotograma.
    """
    seq = np.asarray(sequence, dtype=np.float64).copy()
    if seq.ndim != 2 or seq.shape[1] not in (5, 6) or not len(seq):
        raise ValueError("Se esperaba una secuencia no vacia (N, 5 o 6).")
    if not np.isfinite(seq).all() or not np.isfinite(hand_scale) or hand_scale <= 1e-6:
        raise ValueError("La secuencia y la escala de mano deben ser validas y finitas.")
    seq[:, 0] -= seq[0, 0]
    if np.any(np.diff(seq[:, 0]) <= 0):
        raise ValueError("Los tiempos deben ser estrictamente crecientes.")
    origin = seq[0, 3:5].copy()
    seq[:, 1:3] = (seq[:, 1:3] - origin) / hand_scale
    seq[:, 3:5] = (seq[:, 3:5] - origin) / hand_scale
    return seq


class MotionBuffer:
    """Buffer circular con los ultimos N frames detectados de la mano.

    Guarda los 21 landmarks de cada frame (NO filtra por letra: asi el
    problema I/J siempre acumula fotogramas) y de ahi deriva:

    * ``get_sequence(letra)``  -> [t, tip, wrist, orient] para esa letra;
    * ``is_moving()``          -> si hubo movimiento real (punta, orientacion
      de la mano o traslacion de la muneca), para no invocar al clasificador
      con la mano quieta.
    """

    def __init__(self, maxlen: int = MOTION_BUFFER_SIZE):
        self.maxlen = maxlen
        # Cada entrada: (timestamp, landmarks_xy (21, 2), handedness)
        self._buffer: deque[tuple[float, np.ndarray, str]] = deque(maxlen=maxlen)

    def add(self, detection: HandDetection) -> None:
        """Agrega una deteccion al buffer (sin filtro por letra)."""
        if detection is None:
            return
        self._buffer.append((
            time.monotonic(),
            detection.landmarks[:, :2].astype(np.float64),
            detection.handedness,
        ))

    def clear(self) -> None:
        self._buffer.clear()

    def is_ready(self, min_frames: int = MOTION_MIN_FRAMES) -> bool:
        return len(self._buffer) >= min_frames

    def get_sequence(self, letter: str) -> Optional[np.ndarray]:
        """Secuencia (N, 6) [t, tip_x, tip_y, wrist_x, wrist_y, orient].

        Coordenadas en unidades de mano y segundos desde el primer frame.
        La orientacion es el angulo del vector muneca -> MCP del dedo medio.
        """
        if len(self._buffer) < MOTION_MIN_FRAMES:
            return None
        finger_idx = config.MOTION_FINGER_TIP.get(letter)
        if finger_idx is None:
            return None
        rows = []
        for t, lm, _ in self._buffer:
            wrist = lm[_WRIST]
            orient = np.arctan2(
                lm[_MCP_MIDDLE][1] - wrist[1], lm[_MCP_MIDDLE][0] - wrist[0]
            )
            tip = lm[finger_idx]
            rows.append([t, tip[0], tip[1], wrist[0], wrist[1], orient])
        scale = float(np.mean([
            np.linalg.norm(lm[_MCP_MIDDLE] - lm[_WRIST])
            for _, lm, _ in self._buffer
        ]))
        if scale <= 1e-6:
            return None
        return normalize_motion_sequence(rows, hand_scale=scale)

    def movement_score(self, letter: str | None = None) -> float:
        """Puntaje de movimiento en unidades de tamaño de mano.

        Combina las tres señales (punta, orientación, muñeca); sirve para
        calibrar ``config.MOTION_MIN_MOVEMENT`` y para depuración.
        """
        n = len(self._buffer)
        if n < MOTION_MIN_FRAMES:
            return 0.0

        scales = []
        for _, lm, _ in self._buffer:
            d = float(np.linalg.norm(lm[_MCP_MIDDLE] - lm[_WRIST]))
            scales.append(max(d, 1e-6))
        scale = float(np.mean(scales))

        # 1) puntas relativas a la muneca (o la punta de una letra si se pide)
        tips = (config.MOTION_FINGER_TIP.get(letter),) if letter in config.MOTION_FINGER_TIP else _ALL_FINGER_TIPS
        max_diag = 0.0
        for tip_idx in tips:
            rel = np.array(
                [lm[tip_idx] - lm[_WRIST] for _, lm, _ in self._buffer]
            )
            diag = float(np.linalg.norm(rel.max(axis=0) - rel.min(axis=0)))
            max_diag = max(max_diag, diag)
        d_tip = max_diag / scale

        # 2) orientacion de la mano (rango en unidades de 90 grados)
        orients = np.unwrap(np.array([
            np.arctan2(lm[_MCP_MIDDLE][1] - lm[_WRIST][1],
                       lm[_MCP_MIDDLE][0] - lm[_WRIST][0])
            for _, lm, _ in self._buffer
        ], dtype=np.float64))
        d_orient = float(orients.max() - orients.min()) / (np.pi / 2)

        # 3) traslacion de la muneca
        wrists = np.array([lm[_WRIST] for _, lm, _ in self._buffer])
        d_wrist = float(np.linalg.norm(wrists.max(axis=0) - wrists.min(axis=0))) / scale

        return max(d_tip, d_orient, d_wrist)

    def is_moving(self, min_movement: float | None = None) -> bool:
        """True si en el buffer hubo movimiento real de la mano.

        Tres senales (cualquiera supera el umbral, medido en unidades de
        tamano de mano = distancia muneca -> MCP del dedo medio):

        * desplazamiento de cualquier punta relativa a la muneca
          (trazos de J, Z, X);
        * rango de orientacion de la mano (balanceos de K, Q, Ñ);
        * traslacion de la muneca en la imagen.

        Con la mano quieta devuelve False; la app no confirma letras dinamicas.
        """
        if min_movement is None:
            min_movement = config.MOTION_MIN_MOVEMENT
        if len(self._buffer) < MOTION_MIN_FRAMES:
            return False
        return self.movement_score() >= min_movement

    def get_handedness(self) -> Optional[str]:
        if not self._buffer:
            return None
        # Usar el handedness mas frecuente en el buffer
        handednesses = [h for _, _, h in self._buffer]
        return max(set(handednesses), key=handednesses.count)


# ----------------------------------------------------------------------
# Decision hibrida: ¿cuando usar el clasificador de movimiento?
# ----------------------------------------------------------------------
def candidate_for(static_label: Optional[str]) -> Optional[str]:
    """Letra dinamica cuyo trazo hay que analizar para una etiqueta estatica.

    ``I -> J``, ``N -> Ñ``, ``G -> Q``; las propias dinamicas se mapean a
    si mismas; cualquier otra letra no tiene trazo (devuelve None).
    """
    if not static_label:
        return None
    if static_label in config.MOTION_CANDIDATE_FOR_STATIC:
        return config.MOTION_CANDIDATE_FOR_STATIC[static_label]
    if static_label in config.MOTION_CLASSES:
        return static_label
    return None


def should_use_motion(
    static_label: Optional[str],
    buffer: MotionBuffer,
    min_frames: int = MOTION_MIN_FRAMES,
    min_movement: float | None = None,
) -> bool:
    """Regla hibrida: activar el clasificador de movimiento.

    Solo si: el buffer esta listo, la etiqueta estatica es candidata
    (las 6 dinamicas + I, N, G) Y la mano se esta moviendo de verdad.
    Con la mano quieta no se autoriza ninguna prediccion dinamica.
    """
    if buffer is None or not buffer.is_ready(min_frames):
        return False
    if candidate_for(static_label) is None:
        return False
    if static_label not in config.MOTION_TRIGGER:
        return False
    return buffer.is_moving(min_movement)


# ----------------------------------------------------------------------
# Extraccion de features de trayectoria
# ----------------------------------------------------------------------
def extract_trajectory_features(sequence: np.ndarray) -> np.ndarray:
    """Extrae features de una secuencia de movimiento.
    
    Entrada:
        sequence: array (N, 6) con columnas
        [t, tip_x, tip_y, wrist_x, wrist_y, orient], en unidades de mano.
        Con 5 columnas se asume orientacion ausente y se rellena en ceros.
    
    Salida:
        vector de features (MOTION_NUM_FEATURES,)
    """
    if sequence is None or len(sequence) < MOTION_MIN_FRAMES:
        return np.zeros(config.MOTION_NUM_FEATURES, dtype=np.float32)
    sequence = normalize_motion_sequence(sequence)
    if sequence.shape[1] == 5:
        # Compatibilidad con secuencias antiguas sin orientacion
        pad = np.zeros((sequence.shape[0], 1), dtype=np.float64)
        sequence = np.hstack([sequence, pad])
    
    # Coordenadas relativas a la muneca (invariante a posicion de la mano)
    tip_rel = sequence[:, 1:3] - sequence[:, 3:5]  # (N, 2)
    
    # Segundos relativos en float64: no perder intervalos con relojes grandes.
    t = sequence[:, 0]
    
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
    if np.all(std_xy > 1e-12):
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
    
    # 10. Orientacion de la mano y traslacion de la muneca
    # (K, Ñ y Q se mueven con la muneca: sin esto sus trazos serian
    # invisibles para el clasificador)
    orient = np.unwrap(sequence[:, 5].astype(np.float64))  # des-envolver ±pi
    features.append(float(orient.max() - orient.min()))    # rango de giro (1)
    features.append(float(np.std(orient)))                 # dispersión (1)
    features.append(float(orient[-1] - orient[0]))         # giro neto con signo (1)
    orient_dt = np.diff(orient)
    orient_dt = orient_dt / np.maximum(np.diff(t), 1e-6)
    features.append(float(np.mean(np.abs(orient_dt))))     # vel. angular media (1)
    wrist_xy = sequence[:, 3:5]
    wrist_range = wrist_xy.max(axis=0) - wrist_xy.min(axis=0)
    features.extend(float(v) for v in wrist_range)         # traslacion x,y (2)
    
    # Total: 23 originales + 6 nuevas = 29 features
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
    """Clasificador ligero para las 6 letras dinamicas (J, K, Ñ, Q, X, Z)."""
    
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
            raise ValueError("Se necesitan al menos 2 clases dinamicas para entrenar.")
        
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
            print(f"MotionClassifier ({self.model_type}) - {len(self.classes_)} clases:")
            print(f"  CV accuracy: {np.mean(cv_scores):.3f} (+-{np.std(cv_scores):.3f})")
            print(f"  Test accuracy: {test_acc:.3f}")
            print(report)
        
        return self.metrics_
    
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predice la letra dinamica a partir de features de trayectoria.
        
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
            "feature_version": config.MOTION_FEATURE_VERSION,
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
        if bundle.get("feature_version") != config.MOTION_FEATURE_VERSION:
            raise ValueError("Modelo de movimiento antiguo: reentrena con python src/train.py --motion.")
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
MOTION_CSV_COLUMNS = ["label", "persona", "feature_version"] + MOTION_FEATURE_COLUMNS


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
        "feature_version": config.MOTION_FEATURE_VERSION,
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
        existing = load_motion_dataset(path)
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
    if "feature_version" not in df or not df["feature_version"].eq(config.MOTION_FEATURE_VERSION).all():
        raise ValueError(
            "Dataset de movimiento antiguo o incompatible: vuelve a capturar las "
            "secuencias reales o regenera las sinteticas con scripts/make_sample_motion_data.py."
        )
    
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
    """Detecta si un CSV tiene el esquema de movimiento (m0..m28 + label)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    # Reconocer tambien CSV antiguos para que la carga explique que deben
    # recapturarse, en lugar de ignorarlos y entrenar con datos sinteticos.
    return {"label", "persona", *MOTION_FEATURE_COLUMNS} <= cols


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

    Valida el esquema de cada archivo y las etiquetas (solo letras
    dinamicas: J, K, Ñ, Q, X, Z) antes de guardar.
    """
    if not paths:
        raise ValueError("No hay CSV de movimiento para unir.")
    frames = []
    for p in paths:
        if not is_motion_csv(p):
            raise ValueError(f"'{Path(p).name}' no es un CSV de movimiento.")
        frames.append(load_motion_dataset(p))
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
    """Entrena y guarda el clasificador de movimiento (J, K, Ñ, Q, X, Z).
    
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
