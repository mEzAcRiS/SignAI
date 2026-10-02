"""Genera el dataset de PRUEBA sintetico (data/samples/hand_landmarks_sample.csv).

Importante: estos datos son SINTETICOS y existen solo para que el proyecto
se pueda ejecutar, entrenar y verificar sin una camara (por ejemplo, en la
computadora del docente). El dataset real se recolecta con la camara
usando src/collect_data.py.

Cada muestra se construye como una mano de 21 puntos con la postura de
cada letra del abecedario LSM, se agrega ruido y variacion de
angulo/escala/espejo, y finalmente se normaliza con la misma funcion que
usa el programa en tiempo real (origen en la muneca, escala por tamano).

Uso:
    python scripts/make_sample_data.py
    python scripts/make_sample_data.py --per-class 200 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from dataset import save_dataset, make_sample_row  # noqa: E402
from hand_detector import normalize_landmarks  # noqa: E402

# ---------------------------------------------------------------------------
# Geometria basica de una mano (coordenadas de imagen: y crece hacia abajo)
# ---------------------------------------------------------------------------
BASE_POINTS = np.array([
    [-0.35, -0.10],   # 1  pulgar  CMC
    [-0.28, -0.55],   # 5  indice  MCP
    [-0.02, -0.64],   # 9  medio   MCP
    [0.22, -0.58],    # 13 anular  MCP
    [0.44, -0.48],    # 17 meñique MCP
])

DIRECTIONS = np.array([
    [-0.60, -0.80],   # pulgar (hacia arriba-izquierda)
    [-0.08, -1.00],   # indice
    [0.00, -1.00],    # medio
    [0.06, -1.00],    # anular
    [0.14, -1.00],    # meñique
])

LENGTHS = [
    [0.26, 0.20, 0.18],   # pulgar
    [0.28, 0.20, 0.16],   # indice
    [0.30, 0.22, 0.17],   # medio
    [0.27, 0.19, 0.15],   # anular
    [0.22, 0.15, 0.12],   # meñique
]

# ---------------------------------------------------------------------------
# Postura de cada letra del abecedario LSM (datos sinteticos de prueba)
# ---------------------------------------------------------------------------
# Cada letra define:
#   curl = angulo de doblado por articulacion (grados; 0 = dedo estirado)
#   fan  = giro de la direccion base del dedo (grados; abre/cierra el dedo)
# Orden de los valores: [pulgar, indice, medio, anular, menique]
#
# Son aproximaciones REPRESENTATIVAS de las letras de la dactilologia LSM
# (ver src/letters.py y docs/abecedario_lsm.md), disenadas ademas para ser
# distinguibles entre si: los datos reales se capturan con la camara.
LETTER_POSTURES: dict[str, dict[str, list[float]]] = {
    "A": {"curl": [15, 70, 70, 70, 70], "fan": [30, 0, 0, 0, 0]},
    "B": {"curl": [80, 0, 0, 0, 0], "fan": [0, 0, 0, 0, 0]},
    "C": {"curl": [35, 40, 40, 40, 40], "fan": [0, 0, 0, 0, 0]},
    "D": {"curl": [60, 0, 75, 75, 75], "fan": [0, 0, 0, 0, 0]},
    "E": {"curl": [55, 80, 80, 80, 80], "fan": [10, 0, 0, 0, 0]},
    "F": {"curl": [45, 65, 0, 0, 0], "fan": [0, 0, 0, 0, 0]},
    "G": {"curl": [0, 0, 75, 75, 75], "fan": [-25, -70, 0, 0, 0]},
    "H": {"curl": [70, 0, 0, 75, 75], "fan": [0, -70, -70, 0, 0]},
    "I": {"curl": [65, 75, 75, 75, 0], "fan": [0, 0, 0, 0, 0]},
    "J": {"curl": [65, 75, 75, 75, 35], "fan": [0, 0, 0, 0, 0]},
    "K": {"curl": [0, 0, 0, 75, 75], "fan": [15, -12, 12, 0, 0]},
    "L": {"curl": [0, 0, 75, 75, 75], "fan": [-55, 0, 0, 0, 0]},
    "M": {"curl": [85, 60, 60, 60, 78], "fan": [0, 0, 0, 0, 0]},
    "N": {"curl": [85, 60, 60, 78, 78], "fan": [0, 0, 0, 0, 0]},
    "O": {"curl": [55, 60, 60, 60, 60], "fan": [0, 0, 0, 0, 0]},
    "P": {"curl": [20, 0, 0, 75, 75], "fan": [120, 180, 180, 0, 0]},
    "Q": {"curl": [0, 0, 75, 75, 75], "fan": [140, 150, 0, 0, 0]},
    "R": {"curl": [85, 15, 15, 75, 75], "fan": [0, 8, -8, 0, 0]},
    "S": {"curl": [85, 72, 72, 72, 72], "fan": [35, 0, 0, 0, 0]},
    "T": {"curl": [30, 55, 70, 70, 70], "fan": [45, 0, 0, 0, 0]},
    "U": {"curl": [85, 0, 0, 75, 75], "fan": [0, 4, -4, 0, 0]},
    "V": {"curl": [85, 0, 0, 75, 75], "fan": [0, -20, 20, 0, 0]},
    "W": {"curl": [85, 0, 0, 0, 75], "fan": [0, -22, 0, 22, 0]},
    "X": {"curl": [85, 45, 75, 75, 75], "fan": [0, 0, 0, 0, 0]},
    "Y": {"curl": [0, 75, 75, 75, 0], "fan": [-45, 0, 0, 0, 0]},
    "Z": {"curl": [85, 15, 75, 75, 75], "fan": [45, 15, 0, 0, 0]},
}

WRIST = np.array([0.0, 0.0])


def _rotate(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _finger_chain(base, direction, lengths, curl_deg, rng) -> np.ndarray:
    """Devuelve 4 puntos (base + 3 articulaciones) de un dedo."""
    curl = np.deg2rad(max(0.0, curl_deg + rng.normal(0, 5)))
    out = [base.copy()]
    pos = base.copy()
    dirv = direction / np.linalg.norm(direction)
    for length in lengths:
        dirv = _rotate(dirv, curl)
        pos = pos + dirv * length
        out.append(pos.copy())
    return np.array(out)


def make_hand(label: str, rng: np.random.Generator) -> np.ndarray:
    """Construye una mano (21, 2) con la postura de la letra indicada."""
    spec = LETTER_POSTURES[label]
    curls, fans = spec["curl"], spec["fan"]

    directions = []
    for i in range(5):
        fan = fans[i] + rng.normal(0, 3)   # variacion del abanico por captura
        directions.append(_rotate(DIRECTIONS[i], np.deg2rad(fan)))

    parts = [WRIST.reshape(1, 2)]
    for i in range(5):
        parts.append(
            _finger_chain(BASE_POINTS[i], directions[i], LENGTHS[i], curls[i], rng)
        )
    hand = np.vstack(parts)  # (21, 2)

    # Variacion de angulo y escala por captura
    angle = np.deg2rad(rng.normal(0, 8))
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    hand = hand @ rot.T
    hand *= rng.uniform(0.9, 1.1)

    # Ruido de deteccion (aun con la muneca fija despues de normalizar)
    hand = hand + rng.normal(0, 0.012, hand.shape)
    hand[0] = 0.0

    # Mano izquierda (espejo en x) ~50% de las muestras
    if rng.random() < 0.5:
        hand[:, 0] *= -1.0

    # Profundidad z simulada (pequena, como en MediaPipe)
    z = rng.normal(0, 0.02, size=(21, 1))
    return np.hstack([hand, z])


def generate(per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in config.CLASSES:
        for _ in range(per_class):
            landmarks = make_hand(label, rng)
            features = normalize_landmarks(landmarks)
            rows.append(
                make_sample_row(
                    features, label, person="sintetico", handedness="Right"
                ).iloc[0]
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera el dataset sintetico de prueba.")
    parser.add_argument("--per-class", type=int, default=150, help="muestras por clase")
    parser.add_argument("--seed", type=int, default=42, help="semilla aleatoria")
    parser.add_argument(
        "--out", default=str(config.SAMPLE_DATASET_PATH), help="archivo de salida"
    )
    args = parser.parse_args(argv)

    df = generate(args.per_class, args.seed)
    out = save_dataset(df, args.out)
    print(f"Dataset sintetico generado: {out}")
    print(f"  filas: {len(df)} | clases: {sorted(df[config.LABEL_COLUMN].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
