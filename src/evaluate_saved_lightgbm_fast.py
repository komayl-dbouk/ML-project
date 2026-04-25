from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

from inference_engine import EXPECTED_COLUMNS, TARGET_COL, predict_hierarchical_dataframe


def predict_in_chunks(model, X: pd.DataFrame, chunk_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        print(f"Predicting rows {start:,}-{stop:,}")
        if isinstance(model, dict) and model.get("model_type") == "hierarchical":
            chunk_pred = predict_hierarchical_dataframe(X.iloc[start:stop], model)["predicted_class"].to_numpy()
        else:
            chunk_pred = model.predict(X.iloc[start:stop])
        chunks.append(chunk_pred)
    return np.concatenate(chunks)


def predict_proba_in_chunks(model, X: pd.DataFrame, chunk_size: int) -> np.ndarray | None:
    if isinstance(model, dict) and model.get("model_type") == "hierarchical":
        return None
    if not hasattr(model, "predict_proba"):
        return None

    chunks = []
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        print(f"Predicting probabilities rows {start:,}-{stop:,}")
        chunks.append(model.predict_proba(X.iloc[start:stop]))
    return np.vstack(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast chunked evaluation for saved project models.")
    parser.add_argument("--model-path", type=Path, default=Path("models/archive_lightgbm.joblib"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/raw/CICIoT2023/CICIOT23/test/test.csv"))
    parser.add_argument("--rare-support-threshold", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    print("Loading model:", args.model_path)
    model = joblib.load(args.model_path)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "n_jobs"):
                step.n_jobs = 1

    print("Loading test CSV:", args.test_csv)
    df = pd.read_csv(args.test_csv, low_memory=False)
    X = df[EXPECTED_COLUMNS].astype("float32")
    y = df[TARGET_COL]

    y_pred = predict_in_chunks(model, X, args.chunk_size)
    y_prob = predict_proba_in_chunks(model, X, args.chunk_size)

    labels, support = np.unique(y, return_counts=True)
    rare_labels = labels[support <= args.rare_support_threshold]
    rare_precision = rare_recall = rare_f1 = rare_auprc = macro_auprc = float("nan")
    if len(rare_labels):
        p, r, f, _ = precision_recall_fscore_support(
            y,
            y_pred,
            labels=rare_labels,
            zero_division=0,
        )
        rare_precision = float(np.mean(p))
        rare_recall = float(np.mean(r))
        rare_f1 = float(np.mean(f))

    if y_prob is not None:
        classes = np.asarray(model.classes_)
        y_true_bin = label_binarize(y, classes=classes)
        macro_auprc = float(average_precision_score(y_true_bin, y_prob, average="macro"))

        if len(rare_labels):
            ap_values = []
            for rare_label in rare_labels:
                label_idx = int(np.where(classes == rare_label)[0][0])
                ap_values.append(float(average_precision_score(y_true_bin[:, label_idx], y_prob[:, label_idx])))
            rare_auprc = float(np.mean(ap_values))

    print("\n=== SAVED MODEL OFFICIAL TEST FAST EVAL ===")
    print("Accuracy    :", round(float(accuracy_score(y, y_pred)), 6))
    print("Macro-Prec  :", round(float(precision_score(y, y_pred, average="macro", zero_division=0)), 6))
    print("Macro-F1    :", round(float(f1_score(y, y_pred, average="macro")), 6))
    print("Weighted-F1 :", round(float(f1_score(y, y_pred, average="weighted")), 6))
    print("Macro AUPRC :", round(macro_auprc, 6))
    print(f"Rare mean precision (support <= {args.rare_support_threshold}):", round(rare_precision, 6))
    print(f"Rare mean recall (support <= {args.rare_support_threshold}):", round(rare_recall, 6))
    print(f"Rare mean F1 (support <= {args.rare_support_threshold}):", round(rare_f1, 6))
    print(f"Rare mean AUPRC (support <= {args.rare_support_threshold}):", round(rare_auprc, 6))
    print("Rare labels:", rare_labels.tolist())


if __name__ == "__main__":
    main()
