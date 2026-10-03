"""SignAI - aplicacion de reconocimiento del abecedario LSM (A-Z) en tiempo real.

Entrada:    camara web, archivo de video, imagen o CSV de caracteristicas.
Proceso:    MediaPipe HandLandmarker (21 landmarks -> 63 valores) y
            clasificador Random Forest entrenado con src/train.py.
Salida:     seña identificada en pantalla con confianza, FPS y esqueleto
            de la mano dibujado.

Uso:
    python src/app.py                       # camara web (tiempo real)
    python src/app.py --demo video.mp4      # video sin camara
    python src/app.py --demo-image foto.jpg # imagen estatica
    python src/app.py --csv data/samples/hand_landmarks_sample.csv  # sin video
    python src/app.py --no-gui              # sin ventanas (solo consola)

Teclas: ESC = salir | P = pausa | S = guardar captura
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

import config
from dataset import DatasetError, load_dataset, split_xy
from hand_detector import HandDetector, draw_hand, draw_label
from train import load_classifier, predict_batch, predict_top
from motion import (
    MotionBuffer,
    MotionClassifier,
    MOTION_BUFFER_SIZE,
    candidate_for,
    extract_trajectory_features,
    load_motion_classifier,
    should_use_motion,
)


# ----------------------------------------------------------------------
class SignSmoother:
    """Suaviza las predicciones para evitar parpadeos en pantalla.

    Mantiene las ultimas N etiquetas con confianza suficiente y solo
    cambia el valor mostrado cuando hay mayoria clara.
    """

    def __init__(
        self,
        window: int = config.SMOOTHING_WINDOW,
        min_votes: int = config.MIN_VOTES,
        threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.window = window
        self.min_votes = min_votes
        self.threshold = threshold
        self.votes: deque[str] = deque(maxlen=window)
        self.current: str | None = None

    def update(self, label: str | None, confidence: float) -> str | None:
        if label is None or confidence < self.threshold:
            self.votes.clear()
            return self.current

        self.votes.append(label)
        if not self.votes:
            return self.current

        counts = Counter(self.votes)
        top_label, top_votes = counts.most_common(1)[0]
        if top_votes >= self.min_votes:
            self.current = top_label
        return self.current

    def reset(self) -> None:
        self.votes.clear()
        self.current = None


# ----------------------------------------------------------------------
def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def run_on_frames(
    frames,
    bundle: dict,
    gui: bool,
    window_name: str = "SignAI - reconocimiento del abecedario LSM (A-Z)",
    save_dir: Path | None = None,
    hold: bool = False,
) -> dict:
    """Procesa un iterador de fotogramas BGR y muestra/conserva el resultado."""
    smoother = SignSmoother(min_votes=1 if hold else config.MIN_VOTES)
    detector = HandDetector(running_mode="video")
    
    # Cargar clasificador de movimiento (opcional)
    motion_clf: MotionClassifier | None = None
    try:
        motion_clf = load_motion_classifier()
        print(f"Clasificador de movimiento cargado: {motion_clf.model_type.upper()}")
    except FileNotFoundError:
        print("Clasificador de movimiento no encontrado (opcional). Usando solo clasificador estatico.")
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el clasificador de movimiento: {e}")
    
    motion_buffer = MotionBuffer(maxlen=MOTION_BUFFER_SIZE)
    
    counter: Counter = Counter()
    confident_count = 0
    total = 0
    fps = 0.0
    last_time = time.perf_counter()

    try:
        for idx, frame in enumerate(frames):
            total += 1
            detection = detector.detect(frame, timestamp_ms=_now_ms())

            label_text = "Sin mano"
            sub_text = "Coloque la mano frente a la camara"
            color = (200, 200, 200)
            final_label = None
            final_confidence = 0.0
            motion_used = False

            if detection is not None:
                draw_hand(frame, detection)
                tops = predict_top(bundle, detection.features)
                raw_label, confidence = tops[0]
                
                # Acumular frames SIEMPRE (sin filtro por letra: asi una J
                # que el estatico lee como I tambien llena el buffer)
                motion_buffer.add(detection)
                
                # Regla hibrida: buffer listo + etiqueta candidata (J,K,Ñ,Q,X,Z
                # e I/N/G) + movimiento real >= umbral (anti falsos positivos)
                use_motion = (
                    motion_clf is not None
                    and should_use_motion(raw_label, motion_buffer)
                )
                
                if use_motion:
                    candidate = candidate_for(raw_label)
                    seq = motion_buffer.get_sequence(candidate) if candidate else None
                    if seq is not None:
                        motion_features = extract_trajectory_features(seq)
                        motion_label, motion_conf = motion_clf.predict(motion_features)
                        # Si el clasificador de movimiento esta muy confiado, usarlo
                        if motion_conf >= config.MOTION_MIN_CONFIDENCE:
                            final_label = motion_label
                            final_confidence = motion_conf
                            motion_used = True
                            # No agregar el raw_label al smoother, usamos el refinado
                        else:
                            # Movimiento no confiado, usar el estatico con suavizado
                            smoothed = smoother.update(raw_label, confidence)
                            if smoothed is not None:
                                final_label = smoothed
                                final_confidence = confidence
                    else:
                        smoothed = smoother.update(raw_label, confidence)
                        if smoothed is not None:
                            final_label = smoothed
                            final_confidence = confidence
                else:
                    # Clasificacion estatica normal con suavizado
                    smoothed = smoother.update(raw_label, confidence)
                    if smoothed is not None:
                        final_label = smoothed
                        final_confidence = confidence
                
                counter[raw_label] += 1
                if confidence >= config.CONFIDENCE_THRESHOLD:
                    confident_count += 1

                # Top-3 de predicciones (con 26 clases ayuda a ver alternativas)
                sub_text = "  ".join(f"{l}:{c:.0%}" for l, c in tops)
                if motion_used:
                    sub_text += f"  |  MOTION: {final_label}({final_confidence:.0%})"
                if final_label is not None:
                    label_text = f"Sena: {final_label}"
                    color = (0, 255, 0) if final_confidence >= config.CONFIDENCE_THRESHOLD else (0, 165, 255)
                else:
                    label_text = "Leyendo..."
                    color = (0, 165, 255)
            else:
                smoother.reset()
                motion_buffer.clear()

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6)) if fps else 1.0 / max(now - last_time, 1e-6)
            last_time = now

            draw_label(frame, label_text, f"{sub_text} | {fps:.0f} FPS", color=color)

            if gui:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key in (ord("p"), ord("P")):
                    cv2.waitKey(0)
                if key in (ord("s"), ord("S")) and save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out = save_dir / f"captura_{idx:04d}.jpg"
                    cv2.imwrite(str(out), frame)
                    print(f"captura guardada: {out}")
            elif total % 60 == 0:
                top = counter.most_common(1)[0][0] if counter else "-"
                print(f"  frame {total:>5} | FPS {fps:5.1f} | sena frecuente: {top}")
    finally:
        detector.close()
        if gui:
            if hold:
                print("Ventana abierta: pulsa cualquier tecla para cerrar...")
                cv2.waitKey(0)
            cv2.destroyAllWindows()

    return {
        "frames": total,
        "con_confianza": confident_count,
        "distribucion": dict(counter.most_common()),
    }


# ----------------------------------------------------------------------
def camera_frames(camera_index: int, width: int, height: int):
    """Generador de fotogramas desde la camara web."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara #{camera_index}. "
            "Verifica que este conectada y libre, o usa un modo sin camara: "
            "--demo video.mp4 | --demo-image foto.jpg | --csv datos.csv"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def video_frames(video_path: str | Path):
    """Generador de fotogramas desde un archivo de video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video '{video_path}'.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# ----------------------------------------------------------------------
def run_csv(csv_path: str | Path, bundle: dict) -> int:
    """Modo sin camara ni ventana: recorre un CSV y muestra el resumen."""
    df = load_dataset(csv_path)
    X, y = split_xy(df)
    smoother = SignSmoother()
    counter: Counter = Counter()

    print(f"Procesando {len(X)} muestras de '{csv_path}' ...")
    labels, confidences = predict_batch(bundle, X)

    correct = 0
    for i, (label, confidence, true_label) in enumerate(zip(labels, confidences, y), start=1):
        smoothed = smoother.update(str(label), confidence) or "-"
        counter[smoothed] += 1
        if label == true_label:
            correct += 1
        if i % 200 == 0 or i == len(X):
            print(f"  {i}/{len(X)} muestras")

    print(f"Exactitud sobre el CSV : {correct / len(X):.1%}")
    print(f"Distribucion suavizada : {dict(counter.most_common())}")
    return 0


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="reconocimiento del abecedario LSM (A-Z) en tiempo real.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", metavar="VIDEO", help="modo demostracion con un video (sin camara)")
    source.add_argument("--demo-image", metavar="IMG", help="modo demostracion con una imagen")
    source.add_argument("--csv", metavar="CSV", help="modo sin camara: procesa un CSV de caracteristicas")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="indice de la camara")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="ancho del video")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="alto del video")
    parser.add_argument("--no-gui", action="store_true", help="no abrir ventanas (solo consola)")
    parser.add_argument("--max-frames", type=int, default=None, help="limitar fotogramas (pruebas)")
    args = parser.parse_args(argv)

    try:
        bundle = load_classifier()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Modelo: {bundle['model_name']} | clases {bundle['classes']} | "
          f"entrenado {bundle['trained_at']}")

    try:
        if args.csv:
            return run_csv(args.csv, bundle)

        if args.demo_image:
            img = cv2.imread(args.demo_image)
            if img is None:
                raise RuntimeError(f"No se pudo leer la imagen '{args.demo_image}'.")
            frames = iter([img])
            gui = not args.no_gui
        elif args.demo:
            frames = video_frames(args.demo)
            gui = not args.no_gui
        else:
            frames = camera_frames(args.camera, args.width, args.height)
            gui = not args.no_gui

        if args.max_frames:
            from itertools import islice

            frames = islice(frames, args.max_frames)

        summary = run_on_frames(
            frames, bundle, gui=gui, save_dir=config.DOCS_DIR,
            hold=args.demo_image is not None,
        )
        print("Resumen:")
        print(f"  fotogramas        : {summary['frames']}")
        print(f"  con confianza     : {summary['con_confianza']}")
        print(f"  distribucion      : {summary['distribucion']}")
        return 0

    except (RuntimeError, DatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
