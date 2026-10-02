"""Prediccion del abecedario LSM (A-Z) sobre imagenes, videos o CSV (sin camara).

Sirve para verificar el modelo sin tiempo real y es la base de varias
pruebas del informe de QA.

Uso:
    python src/predict.py --image data/samples/hand_test.jpg
    python src/predict.py --video ruta/video.mp4 --save out.mp4
    python src/predict.py --csv data/samples/hand_landmarks_sample.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import config
from hand_detector import HandDetector, draw_hand, draw_label
from train import load_classifier, predict_batch, predict_features
from dataset import DatasetError, load_dataset, split_xy


# ----------------------------------------------------------------------
def predict_image(image_path: str | Path, show: bool = False, save: bool = True) -> str | None:
    """Clasifica la mano de una imagen estatica."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: no se pudo leer la imagen '{image_path}'", file=sys.stderr)
        return None

    bundle = load_classifier()
    with HandDetector(running_mode="image") as detector:
        detection = detector.detect(img)

    if detection is None:
        print(f"'{image_path.name}': no se detecto ninguna mano.")
        return None

    label, confidence = predict_features(bundle, detection.features)
    print(
        f"'{image_path.name}': sena = {label} | confianza = {confidence:.1%} | "
        f"mano = {detection.handedness}"
    )

    if save or show:
        out = draw_hand(img.copy(), detection)
        draw_label(
            out,
            f"Sena: {label}",
            f"Confianza: {confidence:.1%}",
            color=(0, 255, 0) if confidence >= config.CONFIDENCE_THRESHOLD else (0, 165, 255),
        )
        if save:
            dst = config.DOCS_DIR / f"pred_{image_path.stem}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), out)
            print(f"  imagen anotada guardada en {dst}")
        if show:
            cv2.imshow("SignAI - prediccion", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return label


# ----------------------------------------------------------------------
def predict_video(
    video_path: str | Path,
    save_path: str | Path | None = None,
    max_frames: int | None = None,
) -> dict:
    """Recorre un video y clasifica cada fotograma con mano detectada."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir el video '{video_path}'", file=sys.stderr)
        return {}

    bundle = load_classifier()
    writer = None
    predictions: list[tuple[str, float]] = []
    frames = 0

    with HandDetector(running_mode="video") as detector:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frames >= max_frames):
                break
            frames += 1

            detection = detector.detect(frame, timestamp_ms=frames * 33)
            if detection is not None:
                label, confidence = predict_features(bundle, detection.features)
                predictions.append((label, confidence))
                draw_hand(frame, detection)
                draw_label(
                    frame,
                    f"Sena: {label}",
                    f"Confianza: {confidence:.1%}",
                    color=(0, 255, 0) if confidence >= config.CONFIDENCE_THRESHOLD else (0, 165, 255),
                )

            if save_path is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20,
                        (w, h),
                    )
                writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    counter = Counter(label for label, _ in predictions)
    summary = {
        "frames": frames,
        "frames_con_mano": len(predictions),
        "confianza_promedio": (
            float(np.mean([c for _, c in predictions])) if predictions else 0.0
        ),
        "distribucion": dict(counter.most_common()),
    }
    print(f"Video: {video_path.name}")
    print(f"  fotogramas procesados : {frames}")
    print(f"  con mano detectada    : {summary['frames_con_mano']}")
    print(f"  confianza promedio    : {summary['confianza_promedio']:.1%}")
    print(f"  distribucion          : {summary['distribucion']}")
    if save_path:
        print(f"  video anotado guardado en {save_path}")
    return summary


# ----------------------------------------------------------------------
def predict_csv(csv_path: str | Path) -> float:
    """Clasifica todas las filas de un CSV; devuelve la exactitud si hay etiquetas."""
    bundle = load_classifier()
    df = load_dataset(csv_path)
    X, y = split_xy(df)

    preds, probs = predict_batch(bundle, X)
    accuracy = float(np.mean(preds == y))
    print(f"CSV: {csv_path}")
    print(f"  muestras             : {len(y)}")
    print(f"  exactitud vs etiqueta: {accuracy:.1%}")
    print(f"  confianza promedio   : {float(np.mean(probs)):.1%}")
    print(f"  distribucion         : {dict(Counter(map(str, preds)).most_common())}")
    return accuracy


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prediccion sin camara (imagen/video/CSV).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="ruta a una imagen con la mano")
    group.add_argument("--video", help="ruta a un video")
    group.add_argument("--csv", help="ruta a un CSV con 63 caracteristicas")
    parser.add_argument("--save", nargs="?", const="annotated.mp4", default=None,
                        help="guardar el video anotado (solo con --video)")
    parser.add_argument("--show", action="store_true", help="mostrar ventana (imagen)")
    parser.add_argument("--max-frames", type=int, default=None, help="limitar fotogramas")
    args = parser.parse_args(argv)

    try:
        if args.image:
            predict_image(args.image, show=args.show)
        elif args.video:
            predict_video(args.video, save_path=args.save, max_frames=args.max_frames)
        else:
            predict_csv(args.csv)
    except (DatasetError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
