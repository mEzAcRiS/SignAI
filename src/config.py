"""Configuración central del proyecto SignAI.

Reúne rutas, clases (dígitos 0-5), dimensiones de los datos y
hiperparámetros del clasificador para que todos los módulos usen
la misma información.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _base_dir() -> Path:
    """Raiz del proyecto.

    - Ejecutando desde codigo fuente: la carpeta del repositorio.
    - Ejecutando el .exe de PyInstaller: la carpeta ``_internal`` del
      programa empaquetado (donde se copian los modelos y datos).
    """
    if getattr(sys, "frozen", False):  # PyInstaller
        return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return Path(__file__).resolve().parent.parent


ROOT = _base_dir()
SRC = ROOT / "src"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
DOCS_DIR = ROOT / "docs"

MODEL_LANDMARKS_PATH = MODELS_DIR / "hand_landmarker.task"
MODEL_CLASSIFIER_PATH = MODELS_DIR / "model.joblib"
DATASET_PATH = DATA_DIR / "hand_landmarks.csv"
SAMPLE_DATASET_PATH = SAMPLES_DIR / "hand_landmarks_sample.csv"

# ---------------------------------------------------------------------------
# Clases del proyecto: digitos estaticos 0-5
# ---------------------------------------------------------------------------
CLASSES: list[str] = ["0", "1", "2", "3", "4", "5"]

# ---------------------------------------------------------------------------
# Datos: 21 landmarks x 3 coordenadas (x, y, z) = 63 valores
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21
NUM_COORDS = 3
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63

# Columnas de caracteristicas en el CSV: x0, y0, z0, ..., x20, y20, z20
FEATURE_COLUMNS: list[str] = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

LABEL_COLUMN = "label"
PERSON_COLUMN = "persona"
HANDEDNESS_COLUMN = "mano"

# ---------------------------------------------------------------------------
# Deteccion de landmarks (MediaPipe HandLandmarker)
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# model_complexity no aplica en Tasks API; se usa el modelo float16 oficial.

# ---------------------------------------------------------------------------
# Captura de video / camara
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# Clasificacion en tiempo real
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60   # prob. minima para mostrar una sena
SMOOTHING_WINDOW = 10         # fotogramas de la ventana de votacion
MIN_VOTES = 6                 # votos minimos de la ventana para cambiar etiqueta

# ---------------------------------------------------------------------------
# Entrenamiento (Random Forest principal + SVM / MLP como alternativas)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SVM_PARAMS: dict = {"C": 10.0, "kernel": "rbf", "random_state": RANDOM_STATE}
MLP_PARAMS: dict = {
    "hidden_layer_sizes": (64, 32),
    "max_iter": 800,
    "random_state": RANDOM_STATE,
}
