"""Genera el dataset sintetico de movimiento para las 6 letras dinamicas.

Crea secuencias sinteticas [t, tip_x, tip_y, wrist_x, wrist_y, orient] con
los trazos caracteristicos de cada letra de la LSM (J, K, Ñ, Q, X, Z) y las
convierte en features con ``extract_trajectory_features``.

Sirve de respaldo cuando aun no hay capturas reales con camara:

    python scripts/make_sample_motion_data.py                  # genera CSV
    python scripts/make_sample_motion_data.py --verify         # + prueba SVM
    python scripts/make_sample_motion_data.py --per-class 80   # mas muestras

Uso despues de generar:
    python src/train.py --motion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from motion import (  # noqa: E402
    MOTION_TRACE_HINTS,
    MotionClassifier,
    extract_trajectory_features,
    make_motion_sample_row,
    split_motion_xy,
)

# Unidades: 1 unidad = tamano de la mano (muneca -> MCP del dedo medio).
# La mano apunta hacia arriba (-y); la punta en reposo se define en estas
# mismas unidades relativas a la muneca.

_N_FRAMES_DEFAULT = 16


def _time_axis(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(14, 21))
    dt = float(rng.uniform(0.05, 0.07))
    return np.arange(n, dtype=np.float64) * dt


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _traj_J(rng: np.random.Generator) -> np.ndarray:
    """Meñique arriba: baja y engancha a la izquierda."""
    t = _time_axis(rng)
    n = len(t)
    u = _smoothstep(np.linspace(0, 1, n))
    down = _smoothstep(np.linspace(0, 1, n) * 2)          # baja rapido
    hook = _smoothstep((np.linspace(0, 1, n) - 0.5) * 2)   # gancho al final
    tip = np.stack([
        0.30 - 0.45 * hook,
        -1.00 + 0.70 * np.minimum(down, 1.0),
    ], axis=1)
    return _pack(t, tip, wrist_dy=0.0, orient_jitter=0.10, rng=rng)


def _traj_Z(rng: np.random.Generator) -> np.ndarray:
    """Índice: zigzag horizontal-diagonal-horizontal."""
    t = _time_axis(rng)
    n = len(t)
    u = np.linspace(0, 1, n)
    seg1 = _smoothstep(u * 3)                        # horizontal derecha
    seg2 = _smoothstep((u - 1 / 3) * 3)              # diagonal abajo-izq
    seg3 = _smoothstep((u - 2 / 3) * 3)              # horizontal derecha
    tip = np.stack([
        0.35 + 0.55 * seg1 - 0.75 * seg2 + 0.65 * seg3,
        -1.00 + 0.45 * seg2,
    ], axis=1)
    return _pack(t, tip, wrist_dy=0.0, orient_jitter=0.12, rng=rng)


def _traj_X(rng: np.random.Generator) -> np.ndarray:
    """Índice en gancho: tracción adentro-afuera (rasgueo)."""
    t = _time_axis(rng)
    n = len(t)
    u = _smoothstep(np.linspace(0, 1, n))
    pull = np.sin(u * np.pi)                          # 0 -> 1 -> 0
    tip = np.stack([
        0.45 - 0.55 * pull,
        -1.05 + 0.35 * pull,
    ], axis=1)
    return _pack(t, tip, wrist_dy=0.0, orient_jitter=0.15, rng=rng)


def _traj_K(rng: np.random.Generator) -> np.ndarray:
    """Config V: balanceo de la muñeca arriba-abajo."""
    t = _time_axis(rng)
    n = len(t)
    u = np.linspace(0, 1, n)
    cycles = rng.uniform(1.3, 1.8)
    wrist_dy = 0.40 * np.sin(2 * np.pi * cycles * u)
    orient = 0.55 * np.sin(2 * np.pi * cycles * u)     # ~±31°
    return _pack(t, tip=None, wrist_dy=wrist_dy, orient=orient, rng=rng)


def _traj_ENYE(rng: np.random.Generator) -> np.ndarray:
    """Config N: oscilación lateral de la muñeca (virgulilla ~)."""
    t = _time_axis(rng)
    n = len(t)
    u = np.linspace(0, 1, n)
    cycles = rng.uniform(1.3, 1.8)
    wrist_dx = 0.45 * np.sin(2 * np.pi * cycles * u)
    orient = 0.25 * np.sin(2 * np.pi * cycles * u + np.pi / 2)
    return _pack(t, tip=None, wrist_dx=wrist_dx, orient=orient, rng=rng)


def _traj_Q(rng: np.random.Generator) -> np.ndarray:
    """Gatillo abajo: pivote/sacudida de la muñeca izq-der."""
    t = _time_axis(rng)
    n = len(t)
    u = np.linspace(0, 1, n)
    cycles = rng.uniform(1.2, 1.7)
    orient = 0.70 * np.sin(2 * np.pi * cycles * u)     # ~±40°
    wrist_dx = 0.15 * np.sin(2 * np.pi * cycles * u)
    tip = np.stack([
        0.35 + 0.10 * np.cos(u * 2 * np.pi),
        0.55 + 0.05 * np.sin(u * 2 * np.pi),
    ], axis=1)
    return _pack(t, tip, wrist_dx=wrist_dx, orient=orient, rng=rng)


def _pack(
    t: np.ndarray,
    tip: np.ndarray | None,
    wrist_dx: float | np.ndarray = 0.0,
    wrist_dy: float | np.ndarray = 0.0,
    orient: float | np.ndarray | None = None,
    orient_jitter: float = 0.10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Arma la secuencia (N, 6) [t, tip, wrist, orient] con ruido."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(t)

    wrist = np.stack([
        0.5 + np.broadcast_to(np.asarray(wrist_dx, dtype=np.float64), (n,)),
        0.5 + np.broadcast_to(np.asarray(wrist_dy, dtype=np.float64), (n,)),
    ], axis=1)

    if orient is None:
        orient = np.full(n, -np.pi / 2)               # dedos arriba
    orient = np.broadcast_to(np.asarray(orient, dtype=np.float64), (n,)).copy()

    if tip is None:
        # Punta en reposo del indice (dentro de la mano), girada con la
        # orientacion: K, N y Q no dibujan con el dedo, mueven la muneca.
        rest = np.array([0.35, -1.05])
        c, s = np.cos(orient + np.pi / 2), np.sin(orient + np.pi / 2)
        rot = np.stack([c * rest[0] - s * rest[1], s * rest[0] + c * rest[1]], axis=1)
        tip = wrist + rot
    else:
        tip = wrist + np.asarray(tip, dtype=np.float64)

    tip = tip + rng.normal(0, 0.02, tip.shape)
    wrist = wrist + rng.normal(0, 0.015, wrist.shape)
    orient = orient + rng.normal(0, orient_jitter * 0.3, n)

    if rng.random() < 0.5:                            # mano izquierda (espejo)
        tip[:, 0] = 1.0 - tip[:, 0]
        wrist[:, 0] = 1.0 - wrist[:, 0]
        orient = np.pi - orient

    return np.stack([
        t, tip[:, 0], tip[:, 1], wrist[:, 0], wrist[:, 1], orient,
    ], axis=1).astype(np.float32)


_GENERATORS = {
    "J": _traj_J,
    "Z": _traj_Z,
    "X": _traj_X,
    "K": _traj_K,
    "Ñ": _traj_ENYE,
    "Q": _traj_Q,
}


def make_dataset(per_class: int, seed: int = 42) -> pd.DataFrame:
    """Genera el DataFrame de movimiento sintetico con ``per_class`` por letra."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for label in config.MOTION_CLASSES:
        gen = _GENERATORS[label]
        for _ in range(per_class):
            seq = gen(rng)
            feats = extract_trajectory_features(seq)
            rows.append(make_motion_sample_row(feats, label, person="sintetico"))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset sintetico de movimiento (6 letras).")
    parser.add_argument("--per-class", type=int, default=50, help="muestras por letra")
    parser.add_argument("--out", default=str(config.SAMPLE_MOTION_DATASET_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="entrena SVM y reporta precision")
    args = parser.parse_args()

    df = make_dataset(args.per_class, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=" * 60)
    print(" DATASET SINTETICO DE MOVIMIENTO")
    print("=" * 60)
    print(f" Salida   : {out}")
    print(f" Muestras : {len(df)} ({args.per_class} x {len(config.MOTION_CLASSES)} letras)")
    for label in config.MOTION_CLASSES:
        print(f"   {label}: {MOTION_TRACE_HINTS[label]}")

    if args.verify:
        X, y = split_motion_xy(df)
        clf = MotionClassifier(model_type="svm")
        metrics = clf.train(X, y, verbose=False)
        print("-" * 60)
        print(f" Verificacion SVM: test {metrics['test_accuracy']:.1%}  "
              f"CV {metrics['cv_accuracy_mean']:.1%} (+-{metrics['cv_accuracy_std']:.1%})")
        print(metrics["classification_report"])
        if metrics["test_accuracy"] < 0.85:
            print(" AVISO: precision baja; revisar separabilidad de los trazos.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
