"""Deteccion de landmarks de mano con MediaPipe Tasks (HandLandmarker).

Este modulo envuelve la API *Tasks* de MediaPipe (la API legacy
``mp.solutions.hands`` fue eliminada a partir de la version 0.10.31) y
convierte los 21 puntos clave de la mano en un vector de 63 valores
normalizados, listo para el clasificador.

Normalizacion (invariancia a posicion y tamano de la mano):
    1. Se resta la posicion de la muneca (landmark 0) a todos los puntos.
    2. Se divide entre la distancia maxima muneca -> punta de dedo,
       de modo que la mano se vea igual este lejos o cerca de la camara.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

import config

# Conexiones utiles para dibujar el esqueleto de la mano
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

# Indices de las puntas de los dedos (pulgar, indice, medio, anular, meñique)
FINGERTIP_IDS = (4, 8, 12, 16, 20)


@dataclass
class HandDetection:
    """Resultado de la deteccion de una mano en un fotograma."""

    landmarks: np.ndarray      # (21, 3) coordenadas normalizadas de imagen
    features: np.ndarray       # (63,) vector de caracteristicas normalizado
    handedness: str            # "Left" o "Right"
    score: float               # confianza de la deteccion [0, 1]

    @property
    def label(self) -> Optional[str]:
        """Clase predicha (la completa el clasificador en app/predict)."""
        return getattr(self, "predicted", None)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Convierte 21 puntos en 63 valores normalizados (origen en la muneca).

    Entrada:
        landmarks: array (21, 3) con coordenadas x, y, z de la mano.
    Salida:
        array (63,) con la mano centrada en la muneca y escalada al
        tamano de la mano.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.shape != (config.NUM_LANDMARKS, config.NUM_COORDS):
        raise ValueError(
            f"Se esperaban {config.NUM_LANDMARKS} puntos x {config.NUM_COORDS} "
            f"coordenadas; se recibio {landmarks.shape}"
        )

    wrist = landmarks[0]
    relative = landmarks - wrist

    # Escala: distancia 2D mayor de la muneca a una punta de dedo
    tips = relative[list(FINGERTIP_IDS), :2]
    scale = float(np.max(np.linalg.norm(tips, axis=1)))
    if scale < 1e-6:  # mano degenerada / sin dedos: evita division por cero
        scale = 1.0

    normalized = relative / scale
    return normalized.reshape(config.NUM_FEATURES).astype(np.float32)


def landmarks_array(hand_landmarks) -> np.ndarray:
    """Convierte la lista de objetos landmark de MediaPipe a array (21, 3)."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float64,
    )


class HandDetector:
    """Detecta manos en fotogramas y extrae el vector de 63 valores.

    Parameters
    ----------
    model_path : str | Path
        Ruta al modelo ``hand_landmarker.task`` de MediaPipe.
    running_mode : str
        ``"video"`` para tiempo real (requiere marcas de tiempo) o
        ``"image"`` para imagenes estaticas.
    max_num_hands : int
        Numero maximo de manos a detectar.
    """

    def __init__(
        self,
        model_path=config.MODEL_LANDMARKS_PATH,
        running_mode: str = "video",
        max_num_hands: int = config.MAX_NUM_HANDS,
    ) -> None:
        mode = RunningMode.VIDEO if running_mode == "video" else RunningMode.IMAGE
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self._running_mode = mode
        self._last_timestamp_ms = -1

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[HandDetection]:
        """Detecta la primera mano de un fotograma BGR (numpy/OpenCV).

        En modo ``video`` se debe pasar una marca de tiempo monotonicamente
        creciente; si no se pasa, se calcula con el reloj interno.
        Devuelve ``None`` si no se detecta mano.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == RunningMode.VIDEO:
            if timestamp_ms is None:
                timestamp_ms = int(time.monotonic() * 1000)
            # MediaPipe exige marcas de tiempo estrictamente crecientes
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        lms = landmarks_array(result.hand_landmarks[0])
        handedness = "Right"
        score = 0.0
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name or "Right"
            score = float(result.handedness[0][0].score)

        return HandDetection(
            landmarks=lms,
            features=normalize_landmarks(lms),
            handedness=handedness,
            score=score,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Libera los recursos del modelo."""
        self._detector.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ----------------------------------------------------------------------
# Utilidades de dibujo (sin la API legacy de MediaPipe)
# ----------------------------------------------------------------------
def draw_hand(frame: np.ndarray, detection: HandDetection, color=(0, 255, 0)) -> np.ndarray:
    """Dibuja el esqueleto de la mano sobre el fotograma."""
    h, w = frame.shape[:2]
    pts = [
        (int(lm[0] * w), int(lm[1] * h)) for lm in detection.landmarks
    ]

    for connection in HAND_CONNECTIONS:
        start, end = int(connection.start), int(connection.end)
        cv2.line(frame, pts[start], pts[end], color, 2, cv2.LINE_AA)

    for point in pts:
        cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)

    return frame


def draw_label(
    frame: np.ndarray,
    text: str,
    sub_text: str = "",
    color=(0, 255, 0),
) -> np.ndarray:
    """Dibuja la etiqueta de la sena en la esquina superior izquierda."""
    cv2.rectangle(frame, (8, 8), (300, 96 if sub_text else 60), (20, 20, 20), -1)
    cv2.putText(
        frame, text, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA
    )
    if sub_text:
        cv2.putText(
            frame, sub_text, (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (220, 220, 220), 1, cv2.LINE_AA,
        )
    return frame
