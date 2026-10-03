"""Entrenamiento y comparacion de clasificadores para el abecedario LSM (A-Z).

Flujo (clasificador estatico):
    1. Carga y valida el dataset de 63 caracteristicas.
    2. Divide en entrenamiento/prueba (estratificado 80/20).
    3. Compara tres modelos con validacion cruzada de 5 pliegues:
         - Random Forest  (modelo principal del proyecto)
         - SVM con kernel RBF
         - MLP pequeno (red neuronal de 2 capas)
    4. Entrena el modelo final (Random Forest por defecto) y lo guarda
       en ``models/model.joblib`` junto con las metricas.

Flujo (clasificador de movimiento para J/Z):
    python src/train.py --motion               # entrena modelo de trazo
    python src/train.py --motion --motion-model rf  # con Random Forest

Uso:
    python src/train.py                        # dataset completo (estatico)
    python src/train.py --data data/samples/hand_landmarks_sample.csv
    python src/train.py --model best           # elige el mejor por CV
    python src/train.py --motion               # entrena modelo de movimiento J/Z
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from dataset import DatasetError, load_dataset, split_xy
from motion import train_motion_model, load_motion_classifier, find_motion_csvs, merge_motion_csvs


# ----------------------------------------------------------------------
# Construccion de modelos
# ----------------------------------------------------------------------
def build_models() -> dict[str, object]:
    """Devuelve los tres candidatos con sus hiperparametros."""
    return {
        "rf": RandomForestClassifier(**config.RF_PARAMS),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                # SVC(probability=True) esta deprecado en sklearn 1.9;
                # CalibratedClassifierCV conserva las probabilidades.
                (
                    "svc",
                    CalibratedClassifierCV(
                        SVC(**config.SVM_PARAMS), ensemble=False
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**config.MLP_PARAMS)),
            ]
        ),
    }


MODEL_NAMES = {"rf": "Random Forest", "svm": "SVM (RBF)", "mlp": "MLP (red neuronal)"}


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------
def train(
    data_path: str | Path,
    model_choice: str = "rf",
    out_path: str | Path = config.MODEL_CLASSIFIER_PATH,
    verbose: bool = True,
) -> dict:
    """Entrena, evalua y guarda el clasificador. Devuelve el bundle con metricas."""
    df = load_dataset(data_path)
    X, y = split_xy(df)

    if len(np.unique(y)) < 2:
        raise DatasetError("Se necesitan al menos 2 clases para entrenar.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    if verbose:
        print(f"Dataset : {data_path}")
        print(f" muestras: {len(X)} | clases: {sorted(set(y))}")
        print(f" train   : {len(X_train)} | test: {len(X_test)}")
        print(f" caracteristicas por muestra: {X.shape[1]}")
        print()

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    results: dict[str, dict] = {}
    models = build_models()
    for key, model in models.items():
        t0 = time.perf_counter()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - t0

        results[key] = {
            "nombre": MODEL_NAMES[key],
            "cv_accuracy_mean": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "train_seconds": round(elapsed, 2),
            "model": model,
            "y_pred": y_pred,
        }
        if verbose:
            print(
                f"[{MODEL_NAMES[key]:<15}] "
                f"CV={np.mean(scores):.3f} (+-{np.std(scores):.3f}) | "
                f"test acc={results[key]['test_accuracy']:.3f} | "
                f"F1={results[key]['test_f1_macro']:.3f} | "
                f"{elapsed:.1f}s"
            )

    if verbose:
        print()

    if model_choice == "best":
        chosen = max(results, key=lambda k: results[k]["cv_accuracy_mean"])
    else:
        chosen = model_choice

    final = results[chosen]
    report = classification_report(y_test, final["y_pred"], digits=3, zero_division=0)
    cm = confusion_matrix(y_test, final["y_pred"], labels=sorted(set(y)))

    if verbose:
        print(f"=== Modelo final: {final['nombre']} ===")
        print(report)
        print("Matriz de confusion (filas=real, columnas=pronostico):")
        header = "      " + " ".join(f"{c:>4}" for c in sorted(set(y)))
        print(header)
        for label, row in zip(sorted(set(y)), cm):
            print(f"  {label:>3} | " + " ".join(f"{v:>4}" for v in row))
        print()

    bundle = {
        "model": final["model"],
        "model_name": final["nombre"],
        "model_key": chosen,
        "classes": sorted(set(y)),
        "feature_columns": list(config.FEATURE_COLUMNS),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(data_path),
        "n_samples": int(len(X)),
        "metrics": {
            "cv_accuracy_mean": final["cv_accuracy_mean"],
            "cv_accuracy_std": final["cv_accuracy_std"],
            "test_accuracy": final["test_accuracy"],
            "test_f1_macro": final["test_f1_macro"],
            "all_models": {
                k: {kk: vv for kk, vv in r.items() if kk not in ("model", "y_pred")}
                for k, r in results.items()
            },
            "confusion_matrix": {
                "labels": sorted(set(y)),
                "matrix": cm.tolist(),
            },
            "classification_report": report,
        },
    }

    # En inferencia n_jobs=-1 agrega ~20 ms de overhead por prediccion
    # (el paralelismo solo conviene durante el entrenamiento).
    model_obj = bundle["model"]
    if hasattr(model_obj, "n_jobs"):
        model_obj.n_jobs = 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # compress=3 mantiene el modelo compacto (objetivo: menos de 5 MB)
    joblib.dump(bundle, out_path, compress=3)

    metrics_path = out_path.with_name("metrics.json")
    metrics_path.write_text(
        json.dumps(bundle["metrics"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _save_confusion_matrix(cm, sorted(set(y)))

    if verbose:
        print(f"Modelo guardado en : {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        print(f"Metricas guardadas : {metrics_path}")

    return bundle


def _save_confusion_matrix(cm: np.ndarray, labels: list[str]) -> Path | None:
    """Guarda la matriz de confusion como imagen para el manual/QA."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Pronostico")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion - abecedario LSM (A-Z)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out = config.DOCS_DIR / "confusion_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
def load_classifier(path: str | Path = config.MODEL_CLASSIFIER_PATH) -> dict:
    """Carga el bundle entrenado (modelo + metadatos)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo '{path}'. Ejecuta primero: python src/train.py"
        )
    return joblib.load(path)


def predict_features(bundle: dict, features: np.ndarray) -> tuple[str, float]:
    """Predice la sena a partir de un vector de 63 valores.

    Devuelve (etiqueta, probabilidad). Se usa una sola llamada a
    predict_proba para minimizar la latencia en tiempo real.
    """
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    labels, confidences = predict_batch(bundle, X)
    return str(labels[0]), float(confidences[0])


def predict_top(
    bundle: dict, features: np.ndarray, k: int = config.TOP_K
) -> list[tuple[str, float]]:
    """Devuelve las k predicciones mas probables [(etiqueta, probabilidad)]."""
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1][: max(1, k)]
    return [(str(model.classes_[i]), float(proba[i])) for i in order]


def predict_batch(bundle: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predice muchas muestras a la vez (para CSVs y videos).

    Devuelve (etiquetas, confianzas), ambos con longitud = filas de X.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    model = bundle["model"]
    proba = model.predict_proba(X)
    idx = proba.argmax(axis=1)
    labels = np.array([str(c) for c in np.asarray(model.classes_)[idx]])
    confidences = proba[np.arange(len(idx)), idx]
    return labels, confidences


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena el clasificador del abecedario LSM (A-Z).")
    parser.add_argument(
        "--data", default=str(config.DATASET_PATH),
        help="Ruta al CSV del dataset (por defecto data/hand_landmarks.csv)",
    )
    parser.add_argument(
        "--model", choices=["rf", "svm", "mlp", "best"], default="rf",
        help="Modelo final a guardar (por defecto rf = Random Forest)",
    )
    parser.add_argument(
        "--out", default=str(config.MODEL_CLASSIFIER_PATH),
        help="Ruta de salida del modelo (.joblib)",
    )
    parser.add_argument(
        "--motion", action="store_true",
        help="Entrenar el clasificador de movimiento para J y Z (usa data/motion_landmarks.csv)",
    )
    parser.add_argument(
        "--motion-model", choices=["svm", "rf"], default="svm",
        help="Tipo de modelo para movimiento (por defecto svm)",
    )
    parser.add_argument(
        "--motion-out", default=str(config.MOTION_MODEL_PATH),
        help="Ruta de salida del modelo de movimiento (.joblib)",
    )
    args = parser.parse_args(argv)

    if args.motion:
        # Entrenar modelo de movimiento
        data_path = Path(config.MOTION_DATASET_PATH)
        if not data_path.exists():
            # Auto-union: si los integrantes subieron sus *_motion.csv,
            # se unen aqui la primera vez (README: "se unen automaticamente").
            collected = find_motion_csvs(config.DATA_DIR / "collected")
            if collected:
                print(
                    f"No existe {data_path}; uniendo {len(collected)} "
                    "CSV de movimiento de data/collected:"
                )
                for p in collected:
                    print(f"  - {p.name}")
                merge_motion_csvs(collected, data_path)
                print(f"Dataset de movimiento creado: {data_path}")
            else:
                print(f"ERROR: no existe el dataset de movimiento '{data_path}'.")
                print("Primero captura datos de movimiento con: python src/collect_data.py --person Nombre --motion")
                return 1
        try:
            train_motion_model(
                data_path=data_path,
                model_choice=args.motion_model,
                out_path=args.motion_out,
            )
        except Exception as exc:
            print(f"ERROR de dataset de movimiento: {exc}", file=sys.stderr)
            return 1
        return 0

    data_path = Path(args.data)
    # Si falta el dataset real, se usan automaticamente los datos de prueba
    # para que el programa siempre pueda ejecutarse (p. ej. sin camara).
    if not data_path.exists() and data_path == config.DATASET_PATH:
        if config.SAMPLE_DATASET_PATH.exists():
            print(
                f"Aviso: no existe {data_path} (dataset real). "
                f"Se usan los datos de prueba: {config.SAMPLE_DATASET_PATH}"
            )
            data_path = config.SAMPLE_DATASET_PATH

    try:
        train(data_path, model_choice=args.model, out_path=args.out)
    except DatasetError as exc:
        print(f"ERROR de dataset: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
