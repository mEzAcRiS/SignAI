"""Recoleccion de datos con camara: captura de manos para el abecedario LSM (A-Z).

Herramienta para que cada integrante capture sus muestras en su propia
laptop y luego se unan todos los CSV con ``--merge``.

Uso (en una laptop CON camara):
    python src/collect_data.py --person Alan
    python src/collect_data.py --person Alan --per-class 100 --interval 250

Teclas:
    A-Z     = seleccionar la letra actual (se muestra como formarla)
    ESPACIO = capturar una muestra
    0       = activar/desactivar captura automatica
    ESC     = guardar y salir

Union de todos los CSV en el dataset final:
    python src/collect_data.py --merge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

import config
from dataset import DatasetError, append_samples, make_sample_row, merge_csvs
from hand_detector import HandDetector, draw_hand, draw_label
from letters import ascii_hint, hint_for


# ----------------------------------------------------------------------
def collection_loop(args) -> None:
    """Bucle interactivo de captura con la camara."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.person}.csv"

    counts = {label: 0 for label in config.CLASSES}
    current_label = config.CLASSES[0]
    auto = args.auto
    last_capture = 0.0
    frames = 0
    hands_detected = 0

    print("=" * 60)
    print(" RECOLECCION DE DATOS - SignAI (abecedario LSM)")
    print("=" * 60)
    print(f" Persona      : {args.person}")
    print(f" Salida       : {out_path}")
    print(f" Meta/letra   : {args.per_class}")
    print(" Teclas: A-Z = letra | ESPACIO = capturar | 0 = auto | ESC = salir")
    print("=" * 60)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"ERROR: no se pudo abrir la camara #{args.camera}. "
            "Prueba otro indice con --camera 1.",
            file=sys.stderr,
        )
        return

    detector = HandDetector(running_mode="video")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: la camara dejo de entregar fotogramas.", file=sys.stderr)
                break
            frames += 1
            detection = detector.detect(frame, timestamp_ms=int(time.monotonic() * 1000))

            captured_now = False
            if detection is not None:
                hands_detected += 1
                draw_hand(frame, detection)

                should_capture = False
                if auto and (time.perf_counter() - last_capture) * 1000 >= args.interval:
                    should_capture = counts[current_label] < args.per_class
                if should_capture:
                    _capture(detection, current_label, args.person, out_path, counts)
                    last_capture = time.perf_counter()
                    captured_now = True

            # Panel de estado
            status = f"Letra actual: {current_label}  ({counts[current_label]}/{args.per_class})"
            mode = "AUTO" if auto else "MANUAL"
            color = (0, 255, 0) if detection is not None else (0, 0, 255)
            draw_label(
                frame,
                status,
                f"modo {mode} | mano: {'OK' if detection else 'NO'} | "
                f"total {sum(counts.values())} | {args.person}",
                color=color,
            )
            if captured_now:
                cv2.circle(frame, (frame.shape[1] - 40, 40), 15, (0, 255, 0), -1)

            # Conteo por letra en dos columnas (26 letras no caben en una)
            half = (len(config.CLASSES) + 1) // 2
            for idx, label in enumerate(config.CLASSES):
                col, row = divmod(idx, half)
                x = 18 + col * 200
                y = 125 + row * 22
                highlight = label == current_label
                text = f"{label}: {counts[label]:>4}"
                cv2.putText(
                    frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255) if highlight else (220, 220, 220),
                    2 if highlight else 1, cv2.LINE_AA,
                )

            # Pista: como formar la letra que se esta capturando
            cv2.rectangle(frame, (8, 424), (frame.shape[1] - 8, 472), (20, 20, 20), -1)
            cv2.putText(
                frame, ascii_hint(current_label), (18, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("SignAI - recoleccion de datos", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                current_label = chr(key).upper()
                print(f"  letra {current_label}: {hint_for(current_label)}")
            elif key == ord(" "):
                if detection is None:
                    print("  no se detecto mano; acerca la mano a la camara")
                else:
                    _capture(detection, current_label, args.person, out_path, counts)
                    last_capture = time.perf_counter()
                    print(
                        f"  capturada letra {current_label} "
                        f"({counts[current_label]}/{args.per_class})"
                    )
            elif key == ord("0"):
                auto = not auto
                print(f"  captura automatica: {'ON' if auto else 'OFF'}")

            if all(counts[label] >= args.per_class for label in config.CLASSES):
                print("  meta alcanzada para todas las clases. Pulsa ESC para guardar.")
                auto = False

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("Resumen de la sesion:")
    for label in config.CLASSES:
        print(f"  letra {label}: {counts[label]}")
    print(f"  total: {sum(counts.values())} muestras en {out_path}")
    print(f"  (fotogramas: {frames}, con mano: {hands_detected})")
    print("Siguiente paso: python src/collect_data.py --merge")


def _capture(detection, label: str, person: str, out_path: Path, counts: dict) -> None:
    """Guarda una muestra en el CSV de la persona."""
    row = make_sample_row(
        detection.features, label, person=person, handedness=detection.handedness
    )
    append_samples(row, out_path)
    counts[label] += 1


# ----------------------------------------------------------------------
def do_merge(out_dir: str | Path, output: str | Path) -> int:
    """Une todos los CSV de data/collected en el dataset final."""
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob("*.csv"))
    if not paths:
        print(
            f"No hay CSV en '{out_dir}'. Primero captura datos con "
            "src/collect_data.py --person TuNombre",
            file=sys.stderr,
        )
        return 1
    print(f"Uniendo {len(paths)} archivo(s):")
    for p in paths:
        print(f"  - {p.name}")
    df = merge_csvs(paths, output)
    print(f"Dataset final: {output} ({len(df)} muestras)")
    counts = df[config.LABEL_COLUMN].value_counts().sort_index()
    print(counts.to_string())
    return 0


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Captura de muestras con camara.")
    parser.add_argument("--person", help="nombre del integrante que captura (ej. Alan)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="indice de camara")
    parser.add_argument("--per-class", type=int, default=100, help="meta de muestras por letra")
    parser.add_argument("--interval", type=int, default=250, help="ms entre capturas automaticas")
    parser.add_argument("--auto", action="store_true", help="iniciar en modo captura automatica")
    parser.add_argument("--out", default=str(config.DATA_DIR / "collected"), help="carpeta de salida")
    parser.add_argument("--merge", action="store_true", help="unir todos los CSV y salir")
    parser.add_argument("--output", default=str(config.DATASET_PATH), help="destino del --merge")
    args = parser.parse_args(argv)

    try:
        if args.merge:
            return do_merge(args.out, args.output)
        if not args.person:
            parser.error("falta --person (ej. --person Alan)")
        collection_loop(args)
        return 0
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
