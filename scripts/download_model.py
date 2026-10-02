"""Descarga el modelo HandLandmarker de MediaPipe si no esta presente.

Uso:
    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> int:
    dst = config.MODEL_LANDMARKS_PATH
    if dst.exists():
        print(f"El modelo ya existe: {dst} ({dst.stat().st_size / 1_000_000:.1f} MB)")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo desde\n  {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, dst)
    except Exception as exc:  # sin internet / URL cambiada
        print(f"ERROR: no se pudo descargar el modelo: {exc}", file=sys.stderr)
        return 1

    print(f"Guardado en {dst} ({dst.stat().st_size / 1_000_000:.1f} MB) - licencia Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
