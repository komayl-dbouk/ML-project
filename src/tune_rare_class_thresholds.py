from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from inference_engine import DEFAULT_MODEL_NAME, load_model
from train_compare_on_archive import (
    EXPECTED_COLUMNS,
    TARGET_COL,
    exclude_overlap_rows,
    load_official_test,
    sample_archive_balanced,
    split_train_val,
)


THRESHOLD_OUTPUT = Path("models/archive_lightgbm_thresholds.json")
ABSTAIN_LABEL = "UNCERTAIN_RARE"


def get_rare_labels(y_test: pd.Series, rare_support_threshold: int) -> list[str]:
    labels, support = np.unique(y_test, return_counts=True)
    rare_mask = support <= rare_support_threshold
    return labels[rare_mask].tolist()


def apply_thresholds(
    probabilities: np.ndarray,
    class_names: np.ndarray,
    thresholds: dict[str, float],
    uncertainty_floor: float,
    *,
    rare_labels: list[str] | None = None,
    abstain_on_rare: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_labels: list[str] = []
    selected_confidences: list[float] = []
    selected_ranks: list[int] = []
    threshold_applied: list[bool] = []
    rare_label_set = set(rare_labels or [])

    for probs in probabilities:
        ranked_indices = probs.argsort()[::-1]
        top_idx = ranked_indices[0]
        chosen_idx = top_idx
        chosen_rank = 1
        abstained = False

        top_class = str(class_names[top_idx])
        top_prob = float(probs[top_idx])
        top_threshold = thresholds.get(top_class)

        if abstain_on_rare and top_class in rare_label_set and top_threshold is not None and top_prob < top_threshold:
            pred_labels.append(ABSTAIN_LABEL)
            selected_confidences.append(top_prob)
            selected_ranks.append(1)
            threshold_applied.append(True)
            continue

        for rank, idx in enumerate(ranked_indices, start=1):
            class_name = str(class_names[idx])
            prob = float(probs[idx])
            threshold = thresholds.get(class_name)
            if threshold is None or prob >= threshold:
                chosen_idx = idx
                chosen_rank = rank
                break

        pred_labels.append(str(class_names[chosen_idx]))
        selected_confidences.append(float(probs[chosen_idx]))
        selected_ranks.append(chosen_rank)
        threshold_applied.append(chosen_rank != 1 or abstained)

    pred = np.array(pred_labels, dtype=object)
    conf = np.array(selected_confidences, dtype="float64")
    rank = np.array(selected_ranks, dtype="int64")
    uncertain = np.logical_or(conf < uncertainty_floor, np.array(threshold_applied, dtype=bool))
    return pred, conf, rank, uncertain


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    rare_labels: list[str],
) -> dict[str, float]:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    if rare_labels:
        precision, recall, _, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=rare_labels,
            zero_division=0,
        )
        rare_precision = float(np.mean(precision))
        rare_recall = float(np.mean(recall))
    else:
        support = np.array([])
        rare_precision = float("nan")
        rare_recall = float("nan")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rare_mean_precision": rare_precision,
        "rare_mean_recall": rare_recall,
        "rare_support_total": int(support.sum()) if len(support) else 0,
    }


def compute_abstention_stats(
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    rare_labels: list[str],
) -> dict[str, float]:
    pred = np.asarray(y_pred, dtype=object)
    truth = np.asarray(y_true, dtype=object)
    abstain_mask = (pred == ABSTAIN_LABEL)
    rare_truth_mask = np.isin(truth, rare_labels)

    return {
        "abstain_rate": float(abstain_mask.mean()),
        "abstain_on_rare_truth_rate": float(abstain_mask[rare_truth_mask].mean()) if rare_truth_mask.any() else 0.0,
        "abstain_on_nonrare_truth_rate": float(abstain_mask[~rare_truth_mask].mean()) if (~rare_truth_mask).any() else 0.0,
    }


def print_metrics(
    title: str,
    metrics: dict[str, float],
    changed_fraction: float | None = None,
    uncertain_fraction: float | None = None,
    abstention_stats: dict[str, float] | None = None,
) -> None:
    print(f"\n=== {title} ===")
    print("Accuracy             :", round(metrics["accuracy"], 6))
    print("Macro-F1             :", round(metrics["macro_f1"], 6))
    print("Weighted-F1          :", round(metrics["weighted_f1"], 6))
    print("Rare mean precision  :", round(metrics["rare_mean_precision"], 6))
    print("Rare mean recall     :", round(metrics["rare_mean_recall"], 6))
    if changed_fraction is not None:
        print("Predictions changed  :", f"{changed_fraction:.4%}")
    if uncertain_fraction is not None:
        print("Uncertain rate       :", f"{uncertain_fraction:.4%}")
    if abstention_stats is not None:
        print("Abstain rate         :", f"{abstention_stats['abstain_rate']:.4%}")
        print("Abstain on rare true :", f"{abstention_stats['abstain_on_rare_truth_rate']:.4%}")
        print("Abstain on non-rare  :", f"{abstention_stats['abstain_on_nonrare_truth_rate']:.4%}")


def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def tune_thresholds_on_validation(
    *,
    probabilities: np.ndarray,
    class_names: np.ndarray,
    y_val: pd.Series,
    rare_labels: list[str],
    min_recall_ratio: float,
    uncertainty_floor: float,
    tuning_mask: np.ndarray | None = None,
) -> dict[str, float]:
    baseline_pred = class_names[np.argmax(probabilities, axis=1)]
    if tuning_mask is None:
        tuning_mask = np.ones(len(y_val), dtype=bool)

    y_val_tune = y_val.iloc[np.asarray(tuning_mask)].reset_index(drop=True)
    probabilities_tune = probabilities[np.asarray(tuning_mask)]
    baseline_pred_tune = baseline_pred[np.asarray(tuning_mask)]

    base_precision, base_recall, _, _ = precision_recall_fscore_support(
        y_val_tune,
        baseline_pred_tune,
        labels=rare_labels,
        zero_division=0,
    )
    baseline_by_label = {
        label: {
            "precision": float(p),
            "recall": float(r),
        }
        for label, p, r in zip(rare_labels, base_precision, base_recall)
    }

    order = sorted(rare_labels, key=lambda label: (baseline_by_label[label]["precision"], -baseline_by_label[label]["recall"]))
    thresholds: dict[str, float] = {}

    top1_idx = np.argmax(probabilities_tune, axis=1)
    top1_labels = class_names[top1_idx]
    top1_probs = probabilities_tune[np.arange(len(probabilities_tune)), top1_idx]

    for label in order:
        label_mask = (top1_labels == label)
        candidate_probs = top1_probs[label_mask]
        if candidate_probs.size < 25:
            continue

        quantiles = np.linspace(0.50, 0.95, 10)
        candidates = sorted({round(float(np.quantile(candidate_probs, q)), 4) for q in quantiles if candidate_probs.size})
        candidates = [0.0] + [c for c in candidates if c > 0.0]

        baseline_recall = baseline_by_label[label]["recall"]
        min_allowed_recall = baseline_recall * min_recall_ratio

        best_threshold = thresholds.get(label, 0.0)
        best_tuple = (-1.0, -1.0, -1.0)

        for candidate_threshold in candidates:
            temp_thresholds = dict(thresholds)
            if candidate_threshold > 0.0:
                temp_thresholds[label] = candidate_threshold
            elif label in temp_thresholds:
                temp_thresholds.pop(label)

            tuned_pred, _, _, _ = apply_thresholds(
                probabilities_tune,
                class_names,
                temp_thresholds,
                uncertainty_floor,
                rare_labels=rare_labels,
                abstain_on_rare=True,
            )
            precision, recall, _, _ = precision_recall_fscore_support(
                y_val_tune,
                tuned_pred,
                labels=[label],
                zero_division=0,
            )
            precision_value = float(precision[0])
            recall_value = float(recall[0])
            if recall_value < min_allowed_recall:
                continue

            candidate_tuple = (
                precision_value,
                fbeta_score(precision_value, recall_value, beta=0.5),
                recall_value,
            )
            if candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_threshold = candidate_threshold

        if best_threshold > 0.0:
            thresholds[label] = best_threshold

    return thresholds


def build_hard_tuning_mask(
    *,
    probabilities: np.ndarray,
    class_names: np.ndarray,
    y_true: pd.Series,
    rare_labels: list[str],
) -> np.ndarray:
    top1_idx = np.argmax(probabilities, axis=1)
    top2_idx = np.argsort(probabilities, axis=1)[:, -2]
    top1_labels = class_names[top1_idx]
    top2_labels = class_names[top2_idx]
    top1_probs = probabilities[np.arange(len(probabilities)), top1_idx]
    top2_probs = probabilities[np.arange(len(probabilities)), top2_idx]
    margins = top1_probs - top2_probs

    rare_truth = np.isin(np.asarray(y_true, dtype=object), rare_labels)
    rare_top1 = np.isin(top1_labels, rare_labels)
    rare_top2 = np.isin(top2_labels, rare_labels)
    low_margin = margins < 0.20

    hard_mask = rare_truth | rare_top1 | rare_top2 | low_margin
    return hard_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune rare-class decision thresholds for the deployed LightGBM model.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Inference-engine model name to tune.")
    parser.add_argument("--archive-dir", type=Path, default=Path("data/archive (2)"), help="Archive CSV folder.")
    parser.add_argument("--max-per-label", type=int, default=20_000, help="Balanced archive cap per label.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction from the sampled archive.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed. Match the training run for reproducibility.")
    parser.add_argument("--rare-support-threshold", type=int, default=500, help="Rare label support threshold on official test.")
    parser.add_argument("--min-recall-ratio", type=float, default=0.85, help="Keep at least this fraction of baseline rare recall on validation.")
    parser.add_argument("--uncertainty-floor", type=float, default=0.50, help="Predictions below this confidence are marked uncertain.")
    parser.add_argument("--abstain-on-rare", action="store_true", help="Abstain instead of falling back when a rare-class top prediction is below its threshold.")
    parser.add_argument(
        "--tuning-scope",
        choices=["full", "hard"],
        default="hard",
        help="Use all validation rows or only hard rare/confusable/low-margin rows when learning thresholds.",
    )
    parser.add_argument("--output", type=Path, default=THRESHOLD_OUTPUT, help="Where to save the threshold JSON.")
    args = parser.parse_args()

    print("Loading model:", args.model_name)
    model = load_model(args.model_name)
    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not support predict_proba; threshold tuning needs calibrated class probabilities.")

    print("Loading official test set.")
    _, X_official_test, y_official_test = load_official_test()
    rare_labels = get_rare_labels(y_official_test, args.rare_support_threshold)
    print("Rare labels:", rare_labels)

    print("\nSampling archive and rebuilding the validation split used for threshold tuning.")
    df_sample = sample_archive_balanced(
        archive_dir=args.archive_dir,
        max_per_label=args.max_per_label,
        seed=float(args.seed) / 100.0,
    )
    df_sample, removed = exclude_overlap_rows(
        df_train_pool=df_sample,
        X_reference=X_official_test,
        label="OFFICIAL TEST",
    )
    print("Rows removed during de-overlap:", removed)

    _, df_val = split_train_val(df_sample, val_frac=args.val_frac, seed=args.seed)
    X_val = df_val[EXPECTED_COLUMNS].astype("float32")
    y_val = df_val[TARGET_COL]
    X_test = X_official_test.astype("float32")

    print("\nScoring baseline probabilities.")
    val_prob = model.predict_proba(X_val)
    test_prob = model.predict_proba(X_test)
    class_names = model.classes_
    tuning_mask = np.ones(len(y_val), dtype=bool)
    if args.tuning_scope == "hard":
        tuning_mask = build_hard_tuning_mask(
            probabilities=val_prob,
            class_names=class_names,
            y_true=y_val,
            rare_labels=rare_labels,
        )
        print("\n=== HARD TUNING SUBSET ===")
        print("Rows in validation:", len(y_val))
        print("Rows used for tuning:", int(tuning_mask.sum()))
        print("Tuning subset share:", f"{float(tuning_mask.mean()):.4%}")

    base_val_pred, _, _, base_val_uncertain = apply_thresholds(val_prob, class_names, {}, args.uncertainty_floor)
    base_test_pred, _, _, base_test_uncertain = apply_thresholds(test_prob, class_names, {}, args.uncertainty_floor)

    base_val_metrics = compute_metrics(y_val, base_val_pred, rare_labels)
    base_test_metrics = compute_metrics(y_official_test, base_test_pred, rare_labels)
    base_val_abstain = compute_abstention_stats(y_true=y_val, y_pred=base_val_pred, rare_labels=rare_labels)
    base_test_abstain = compute_abstention_stats(y_true=y_official_test, y_pred=base_test_pred, rare_labels=rare_labels)

    print_metrics("VALIDATION BASELINE", base_val_metrics, changed_fraction=0.0, uncertain_fraction=float(base_val_uncertain.mean()), abstention_stats=base_val_abstain)
    print_metrics("OFFICIAL TEST BASELINE", base_test_metrics, changed_fraction=0.0, uncertain_fraction=float(base_test_uncertain.mean()), abstention_stats=base_test_abstain)

    print("\nTuning rare-class thresholds on validation.")
    thresholds = tune_thresholds_on_validation(
        probabilities=val_prob,
        class_names=class_names,
        y_val=y_val,
        rare_labels=rare_labels,
        min_recall_ratio=args.min_recall_ratio,
        uncertainty_floor=args.uncertainty_floor,
        tuning_mask=tuning_mask,
    )

    tuned_val_pred, _, tuned_val_rank, tuned_val_uncertain = apply_thresholds(
        val_prob,
        class_names,
        thresholds,
        args.uncertainty_floor,
        rare_labels=rare_labels,
        abstain_on_rare=args.abstain_on_rare,
    )
    tuned_test_pred, _, tuned_test_rank, tuned_test_uncertain = apply_thresholds(
        test_prob,
        class_names,
        thresholds,
        args.uncertainty_floor,
        rare_labels=rare_labels,
        abstain_on_rare=args.abstain_on_rare,
    )

    tuned_val_metrics = compute_metrics(y_val, tuned_val_pred, rare_labels)
    tuned_test_metrics = compute_metrics(y_official_test, tuned_test_pred, rare_labels)
    tuned_val_abstain = compute_abstention_stats(y_true=y_val, y_pred=tuned_val_pred, rare_labels=rare_labels)
    tuned_test_abstain = compute_abstention_stats(y_true=y_official_test, y_pred=tuned_test_pred, rare_labels=rare_labels)

    changed_val = float(np.mean(tuned_val_pred != base_val_pred))
    changed_test = float(np.mean(tuned_test_pred != base_test_pred))

    print_metrics("VALIDATION THRESHOLDED", tuned_val_metrics, changed_fraction=changed_val, uncertain_fraction=float(tuned_val_uncertain.mean()), abstention_stats=tuned_val_abstain)
    print_metrics("OFFICIAL TEST THRESHOLDED", tuned_test_metrics, changed_fraction=changed_test, uncertain_fraction=float(tuned_test_uncertain.mean()), abstention_stats=tuned_test_abstain)

    print("\n=== SELECTED THRESHOLDS ===")
    if thresholds:
        for label, threshold in sorted(thresholds.items(), key=lambda item: item[1], reverse=True):
            print(f"{label:<25} {threshold:.4f}")
    else:
        print("No class-specific thresholds were selected.")

    print("\n=== DEPLOYMENT EFFECT ===")
    print("Validation fallback-to-lower-rank rate:", f"{np.mean(tuned_val_rank > 1):.4%}")
    print("Official test fallback-to-lower-rank rate:", f"{np.mean(tuned_test_rank > 1):.4%}")

    config = {
        "enabled": True,
        "model_name": args.model_name,
        "rare_support_threshold": args.rare_support_threshold,
        "min_recall_ratio": args.min_recall_ratio,
        "uncertainty_floor": args.uncertainty_floor,
        "tuning_scope": args.tuning_scope,
        "fallback_to_next_rank": not args.abstain_on_rare,
        "abstain_on_low_confidence_rare": bool(args.abstain_on_rare),
        "abstain_label": ABSTAIN_LABEL,
        "rare_classes": rare_labels,
        "thresholds": thresholds,
        "metrics": {
            "validation_baseline": base_val_metrics,
            "validation_thresholded": tuned_val_metrics,
            "official_test_baseline": base_test_metrics,
            "official_test_thresholded": tuned_test_metrics,
            "validation_changed_fraction": changed_val,
            "official_test_changed_fraction": changed_test,
            "validation_abstention": tuned_val_abstain,
            "official_test_abstention": tuned_test_abstain,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nThreshold config saved to:", args.output)


if __name__ == "__main__":
    main()
