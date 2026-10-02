"""Pruebas de la extraccion y normalizacion de landmarks (hand_detector)."""

from __future__ import annotations

import numpy as np
import pytest

import config
from hand_detector import normalize_landmarks, landmarks_array, draw_hand, HandDetection


def _fake_hand() -> np.ndarray:
    """Mano sintetica (21, 3): muneca en el origen, dedos hacia arriba."""
    hand = np.zeros((config.NUM_LANDMARKS, config.NUM_COORDS))
    for i in range(1, config.NUM_LANDMARKS):
        hand[i, 1] = -0.1 * i        # y crece hacia arriba (negativo)
        hand[i, 0] = 0.02 * (i % 4)   # separacion lateral
        hand[i, 2] = -0.001 * i       # profundidad leve
    return hand


def test_shape_63_valores():
    features = normalize_landmarks(_fake_hand())
    assert features.shape == (config.NUM_FEATURES,)
    assert features.dtype == np.float32


def test_muneca_en_origen():
    features = normalize_landmarks(_fake_hand())
    assert np.allclose(features[:3], 0.0, atol=1e-6)


def test_escala_invariante():
    """Manos mas grandes o desplazadas producen el mismo vector."""
    base = _fake_hand()
    grande = base * 3.0 + np.array([5.0, -2.0, 0.0])
    a = normalize_landmarks(base)
    b = normalize_landmarks(grande)
    assert np.allclose(a, b, atol=1e-5)


def test_forma_invalida_genera_error():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((10, 3)))


def test_sin_division_por_cero():
    """Todos los puntos iguales (mano degenerada) no debe provocar NaN."""
    hand = np.ones((config.NUM_LANDMARKS, config.NUM_COORDS)) * 0.5
    features = normalize_landmarks(hand)
    assert not np.any(np.isnan(features))


def test_landmarks_array_desde_objetos_mediapipe():
    class _LM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    objs = [_LM(i, i * 2, i * 3) for i in range(config.NUM_LANDMARKS)]
    arr = landmarks_array(objs)
    assert arr.shape == (config.NUM_LANDMARKS, 3)
    assert arr[5, 0] == 5 and arr[5, 1] == 10 and arr[5, 2] == 15


def test_draw_hand_no_modifica_dimensiones():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    detection = HandDetection(
        landmarks=_fake_hand(),
        features=normalize_landmarks(_fake_hand()),
        handedness="Right",
        score=0.9,
    )
    out = draw_hand(frame.copy(), detection)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # dibujo algo
