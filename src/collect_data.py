"""Recoleccion de datos con camara: captura de manos para el abecedario LSM (A-Z).

Herramienta para que cada integrante capture sus muestras en su propia
laptop y luego se unan todos los CSV con ``--merge``.

Uso (en una laptop CON camara):
    python src/collect_data.py --person Cristhian
    python src/collect_data.py --person Cristhian --per-class 100 --interval 250

Teclas (modo estatico):
    A-Z     = seleccionar la letra actual (se muestra como formarla)
    ESPACIO = capturar una muestra
    0       = activar/desactivar captura automatica
    ESC     = guardar y salir

Modo movimiento (para J y Z):
    python src/collect_data.py --person Cristhian --motion
    Teclas: J/Z = seleccionar letra | ESPACIO = iniciar/parar trazo | M = cancelar | ESC = salir

Union de todos los CSV en el dataset final:
    python src/collect_data.py --merge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

import config
from dataset import DatasetError, append_samples, make_sample_row, merge_csvs
from hand_detector import HandDetector, draw_hand, draw_label
from letters import ascii_hint, hint_for
from motion import (
    MotionBuffer,
    MOTION_BUFFER_SIZE,
    MOTION_MIN_FRAMES,
    MOTION_CSV_COLUMNS,
    extract_trajectory_features,
    make_motion_sample_row,
    merge_motion_csvs,
    save_motion_samples,
    MOTION_FEATURE_COLUMNS,
)


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


# ----------------------------------------------------------------------
# Captura de movimiento para J y Z
# ----------------------------------------------------------------------
def motion_collection_loop(args) -> None:
    """Bucle interactivo de captura de trayectorias para J y Z."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.person}_motion.csv"

    counts = {label: 0 for label in config.MOTION_CLASSES}
    current_label = config.MOTION_CLASSES[0]
    target_per_class = args.per_class
    frames = 0
    hands_detected = 0

    print("=" * 60)
    print(" RECOLECCION DE MOVIMIENTO - SignAI (J y Z)")
    print("=" * 60)
    print(f" Persona      : {args.person}")
    print(f" Salida       : {out_path}")
    print(f" Meta/letra   : {target_per_class} secuencias")
    print(" Teclas: J/Z = seleccionar letra | ESPACIO = iniciar captura de trazo")
    print("       M = cancelar captura actual | ESC = salir")
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
    motion_buffer = MotionBuffer(maxlen=MOTION_BUFFER_SIZE)

    # Estado de captura de movimiento
    capturing_motion = False
    motion_start_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: la camara dejo de entregar fotogramas.", file=sys.stderr)
                break
            frames += 1
            detection = detector.detect(frame, timestamp_ms=int(time.monotonic() * 1000))

            # Agregar al buffer si estamos capturando movimiento
            if capturing_motion and detection is not None:
                motion_buffer.add(detection, current_label)

            # Panel de estado
            if capturing_motion:
                status = f"CAPTURANDO TRAZO: {current_label}  ({len(motion_buffer._buffer)}/{MOTION_BUFFER_SIZE})"
                color = (0, 255, 255)  # amarillo
                sub_text = f"Haz el trazo de {current_label}...  M=cancelar  |  mano: {'OK' if detection else 'NO'}"
            else:
                status = f"Letra: {current_label}  ({counts[current_label]}/{target_per_class})"
                color = (0, 255, 0) if detection is not None else (0, 0, 255)
                sub_text = f"ESPACIO=iniciar trazo  J/Z=cambiar  |  mano: {'OK' if detection else 'NO'}  |  {args.person}"

            if detection is not None:
                hands_detected += 1
                draw_hand(frame, detection)
                # Dibujar el dedo relevante para la letra actual
                finger_idx = config.MOTION_FINGER_TIP.get(current_label)
                if finger_idx is not None:
                    h, w = frame.shape[:2]
                    tip = detection.landmarks[finger_idx]
                    cv2.circle(frame, (int(tip[0] * w), int(tip[1] * h)), 8, (0, 255, 255), -1, cv2.LINE_AA)

            draw_label(frame, status, sub_text, color=color)

            # Mostrar conteo de secuencias capturadas
            for idx, label in enumerate(config.MOTION_CLASSES):
                y = 125 + idx * 30
                highlight = label == current_label
                text = f"{label}: {counts[label]:>4} / {target_per_class}"
                cv2.putText(
                    frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if highlight else (220, 220, 220),
                    2 if highlight else 1, cv2.LINE_AA,
                )

            # Pista de como hacer el trazo
            if current_label == "J":
                hint = "J: Menique estirado, traza una J en el aire (gancho hacia adentro)"
            else:
                hint = "Z: Indice estirado, traza una Z en el aire (zigzag horizontal)"
            cv2.rectangle(frame, (8, 424), (frame.shape[1] - 8, 472), (20, 20, 20), -1)
            cv2.putText(
                frame, hint, (18, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("SignAI - recoleccion de movimiento (J/Z)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key in (ord("j"), ord("J")):
                current_label = "J"
                print(f"  letra {current_label}: trazo de J (gancho con menique)")
            elif key in (ord("z"), ord("Z")):
                current_label = "Z"
                print(f"  letra {current_label}: trazo de Z (zigzag con indice)")

            if key == ord(" "):  # ESPACIO - iniciar/parar captura de trazo
                if not capturing_motion:
                    if detection is None:
                        print("  no se detecto mano; acerca la mano a la camara")
                    else:
                        capturing_motion = True
                        motion_buffer.clear()
                        motion_start_time = time.perf_counter()
                        print(f"  >>> INICIANDO captura de trazo para {current_label} <<<")
                else:
                    # Parar captura y guardar si hay suficientes frames
                    capturing_motion = False
                    duration = time.perf_counter() - motion_start_time
                    if motion_buffer.is_ready(MOTION_MIN_FRAMES):
                        seq = motion_buffer.get_sequence()
                        features = extract_trajectory_features(seq)
                        row = make_motion_sample_row(features, current_label, person=args.person)
                        save_motion_samples(row, out_path)
                        counts[current_label] += 1
                        print(f"  >>> Trazo de {current_label} guardado ({counts[current_label]}/{target_per_class}) duracion: {duration:.2f}s <<<")
                    else:
                        print(f"  trazo muy corto ({len(motion_buffer._buffer)} frames), descartado (min {MOTION_MIN_FRAMES})")
                    motion_buffer.clear()

            elif key == ord("m") or key == ord("M"):  # M - cancelar captura actual
                if capturing_motion:
                    capturing_motion = False
                    motion_buffer.clear()
                    print("  captura de trazo cancelada")

            # Auto-guardar si el buffer se llena
            if capturing_motion and len(motion_buffer._buffer) >= MOTION_BUFFER_SIZE:
                capturing_motion = False
                seq = motion_buffer.get_sequence()
                features = extract_trajectory_features(seq)
                row = make_motion_sample_row(features, current_label, person=args.person)
                save_motion_samples(row, out_path)
                counts[current_label] += 1
                print(f"  >>> Buffer lleno - trazo de {current_label} guardado auto ({counts[current_label]}/{target_per_class}) <<<")
                motion_buffer.clear()

            if all(counts[label] >= target_per_class for label in config.MOTION_CLASSES):
                print("  meta alcanzada para J y Z. Pulsa ESC para guardar.")
                # No auto=False aqui porque no hay modo auto en movimiento

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("Resumen de la sesion de movimiento:")
    for label in config.MOTION_CLASSES:
        print(f"  letra {label}: {counts[label]} secuencias")
    print(f"  total: {sum(counts.values())} secuencias en {out_path}")
    print(f"  (fotogramas totales: {frames}, con mano: {hands_detected})")
    print("Siguiente paso: python src/train.py --motion")


def _capture(detection, label: str, person: str, out_path: Path, counts: dict) -> None:
    """Guarda una muestra en el CSV de la persona."""
    row = make_sample_row(
        detection.features, label, person=person, handedness=detection.handedness
    )
    append_samples(row, out_path)
    counts[label] += 1


# ----------------------------------------------------------------------
def do_merge(
    out_dir: str | Path,
    output: str | Path,
    motion_output: str | Path | None = None,
) -> int:
    """Une todos los CSV de data/collected en los datasets finales.

    Clasifica cada CSV por su esquema: los estaticos (63 features) van a
    ``output`` y los de movimiento (m0..m19, J/Z) a ``motion_output``.
    Los archivos que no coinciden con ninguno se ignoran con aviso.
    """
    out_dir = Path(out_dir)
    motion_output = Path(motion_output or config.MOTION_DATASET_PATH)
    paths = sorted(out_dir.glob("*.csv"))
    if not paths:
        print(
            f"No hay CSV en '{out_dir}'. Primero captura datos con "
            "src/collect_data.py --person TuNombre",
            file=sys.stderr,
        )
        return 1

    static_paths: list[Path] = []
    motion_paths: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except Exception:
            skipped.append(p)
            continue
        if set(config.FEATURE_COLUMNS) <= cols:
            static_paths.append(p)
        elif set(MOTION_CSV_COLUMNS) <= cols:
            motion_paths.append(p)
        else:
            skipped.append(p)

    if skipped:
        print(f"CSV ignorados (esquema desconocido): {len(skipped)}")
        for p in skipped:
            print(f"  - {p.name}")

    ok = False
    if static_paths:
        print(f"Uniendo {len(static_paths)} CSV estaticos:")
        for p in static_paths:
            print(f"  - {p.name}")
        df = merge_csvs(static_paths, output)
        print(f"Dataset estatico final: {output} ({len(df)} muestras)")
        counts = df[config.LABEL_COLUMN].value_counts().sort_index()
        print(counts.to_string())
        ok = True

    if motion_paths:
        print(f"Uniendo {len(motion_paths)} CSV de movimiento (J/Z):")
        for p in motion_paths:
            print(f"  - {p.name}")
        mdf = merge_motion_csvs(motion_paths, motion_output)
        print(f"Dataset de movimiento final: {motion_output} ({len(mdf)} muestras)")
        ok = True

    if not ok:
        print("No se encontro ningun CSV valido para unir.", file=sys.stderr)
        return 1
    print("Siguiente paso: python src/train.py  (y --motion si hubo trazos)")
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
    parser.add_argument("--motion", action="store_true", help="modo captura de movimiento para J y Z")
    args = parser.parse_args(argv)

    try:
        if args.merge:
            return do_merge(args.out, args.output)
        if args.motion:
            if not args.person:
                parser.error("falta --person (ej. --person Alan)")
            motion_collection_loop(args)
            return 0
        if not args.person:
            parser.error("falta --person (ej. --person Alan)")
        collection_loop(args)
        return 0
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
