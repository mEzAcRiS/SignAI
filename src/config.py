"""Configuración central del proyecto SignAI.

Reúne rutas, clases (abecedario LSM A-Z), dimensiones de los datos y
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
# Clases del proyecto: abecedario de la LSM (dactilologia A-Z)
# 6 letras (J, K, Ñ, Q, X, Z) son dinamicas: se capturan con su postura
# estatica base y se refinan con el clasificador de movimiento (MOTION_*).
# ---------------------------------------------------------------------------
CLASSES: list[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

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
# Con 26 clases la probabilidad individual tiende a ser menor que con 6:
# se baja el umbral para no perder la letra en pantalla.
CONFIDENCE_THRESHOLD = 0.45   # prob. minima para mostrar una sena
SMOOTHING_WINDOW = 10         # fotogramas de la ventana de votacion
MIN_VOTES = 6                 # votos minimos de la ventana para cambiar etiqueta
TOP_K = 3                     # predicciones alternativas mostradas en pantalla

# ---------------------------------------------------------------------------
# Entrenamiento (Random Forest principal + SVM / MLP como alternativas)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

RF_PARAMS: dict = {
    # n_estimators/max_depth/min_samples_leaf se acotaron para mantener el
    # modelo por debajo de 5 MB (compromiso de la ficha del proyecto) sin
    # perder exactitud en las 26 letras.
    "n_estimators": 150,
    "max_depth": 16,
    "min_samples_leaf": 4,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SVM_PARAMS: dict = {"C": 10.0, "kernel": "rbf", "random_state": RANDOM_STATE}
MLP_PARAMS: dict = {
    "hidden_layer_sizes": (64, 32),
    "max_iter": 800,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Reconocimiento de movimiento: letras dinamicas del abecedario LSM
# ---------------------------------------------------------------------------
# El alfabeto LSM (27 letras) tiene 21 señas estaticas y 6 dinamicas que
# ademas implican movimiento de la mano o la muneca: J, K, Ñ, Q, X, Z.
# Fuentes: dataset del alfabeto LSM (ScienceDirect, 2026, "static and
# dynamic signs for the Mexican Sign Language alphabet") y Manos con voz
# (Fleischmann y Gonzalez Perez, 2011).
MOTION_CLASSES: list[str] = ["J", "K", "Ñ", "Q", "X", "Z"]

# Landmark de la punta del dedo que dibuja/acompaña el trazo de cada letra
# (20 = meñique para J; 8 = indice para el resto).
MOTION_FINGER_TIP: dict[str, int] = {"J": 20, "K": 8, "Ñ": 8, "Q": 8, "X": 8, "Z": 8}

# Etiquetas ESTATICAS que, junto con las dinamicas, disparan el analisis
# de movimiento: I se lee J (mismo meñique), N se lee Ñ (Ñ = N con
# balanceo) y G se lee Q (Q = G hacia abajo con pivote).
MOTION_TRIGGER: list[str] = MOTION_CLASSES + ["I", "N", "G"]

# Equivalencia etiqueta estatica -> letra dinamica cuyo trazo hay que
# analizar con la punta indicada.
MOTION_CANDIDATE_FOR_STATIC: dict[str, str] = {"I": "J", "N": "Ñ", "G": "Q"}

# Tamaño del buffer de frames para capturar la trayectoria
MOTION_BUFFER_SIZE = 20
MOTION_MIN_FRAMES = 10  # minimo frames para features validos

# Umbral de movimiento (en unidades de tamaño de mano): por debajo se
# considera postura quieta y NO se invoca al clasificador de movimiento.
# Evita falsos positivos de las letras dinamicas.
MOTION_MIN_MOVEMENT = 0.30
# Confianza minima del clasificador de movimiento para reemplazar al
# clasificador estatico.
MOTION_MIN_CONFIDENCE = 0.60

# Numero de features de trayectoria extraidas
# (23 originales de la punta + 6 de orientacion/traduccion de la muneca,
# necesarias para K, Ñ y Q que se mueven con la muneca)
MOTION_NUM_FEATURES = 29

# Rutas del modelo y dataset de movimiento
MOTION_DATASET_PATH = DATA_DIR / "motion_landmarks.csv"
MOTION_MODEL_PATH = MODELS_DIR / "motion_model.joblib"
SAMPLE_MOTION_DATASET_PATH = SAMPLES_DIR / "motion_landmarks_sample.csv"

# Modelo de movimiento por defecto
MOTION_MODEL_TYPE = "svm"  # "svm" o "rf"
