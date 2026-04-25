from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
from lightgbm_device import add_lightgbm_device_args, lightgbm_device_params, print_lightgbm_device

from hierarchical_labels import BENIGN_FAMILY, FAMILY_BY_LABEL, add_family_column, families_in_order, labels_for_family
from train_compare_on_archive import (
    EXPECTED_COLUMNS,
    TARGET_COL,
    build_lgbm_fit_kwargs,
    exclude_overlap_rows,
    load_official_test,
    sample_archive_balanced,
    split_train_val,
)


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_PATH = MODELS_DIR / "hierarchical_lightgbm_bundle.joblib"


@dataclass(frozen=True)
class ScoreSummary:
    accuracy: float
    macro_f1: float
    weighted_f1: float


def summarize(y_true: pd.Series, y_pred: np.ndarray) -> ScoreSummary:
    return ScoreSummary(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted")),
    )


def print_summary(title: str, summary: ScoreSummary) -> None:
    print(f"\n=== {title} ===")
    print("Accuracy    :", round(summary.accuracy, 6))
    print("Macro-F1    :", round(summary.macro_f1, 6))
    print("Weighted-F1 :", round(summary.weighted_f1, 6))


def sample_family_capped(df: pd.DataFrame, family_col: str, max_per_family: int, seed: int) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []
    for family, group in df.groupby(family_col, sort=False):
        take_n = min(len(group), max_per_family)
        sampled_parts.append(group.sample(n=take_n, random_state=seed).reset_index(drop=True))
    return pd.concat(sampled_parts, ignore_index=True)


def compute_family_class_weights(y_family: pd.Series) -> dict[str, float]:
    families = np.unique(y_family)
    weights = compute_class_weight(class_weight="balanced", classes=families, y=y_family)
    return dict(zip(families, weights))


def build_family_candidates(
    seed: int,
    n_jobs: int,
    class_weights: dict[str, float],
    num_classes: int,
    device_params: dict | None = None,
) -> list[tuple[str, dict]]:
    device_params = device_params or {}
    common = dict(
        objective="multiclass",
        num_class=num_classes,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight=class_weights,
        verbose=-1,
        **device_params,
    )
    return [
        (
            "family_balanced_baseline",
            dict(
                **common,
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=6,
                min_child_samples=140,
                subsample=0.75,
                colsample_bytree=0.70,
                reg_alpha=0.8,
                reg_lambda=3.0,
            ),
        ),
        (
            "family_stable_medium",
            dict(
                **common,
                n_estimators=1400,
                learning_rate=0.025,
                num_leaves=11,
                max_depth=5,
                min_child_samples=180,
                subsample=0.70,
                colsample_bytree=0.65,
                reg_alpha=1.2,
                reg_lambda=4.0,
                min_split_gain=0.05,
                max_bin=127,
            ),
        ),
        (
            "family_stable_strong",
            dict(
                **common,
                n_estimators=1600,
                learning_rate=0.02,
                num_leaves=9,
                max_depth=4,
                min_child_samples=240,
                subsample=0.65,
                colsample_bytree=0.60,
                reg_alpha=1.5,
                reg_lambda=5.0,
                min_split_gain=0.10,
                max_bin=127,
            ),
        ),
    ]


def family_selection_score(train_summary: ScoreSummary, val_summary: ScoreSummary) -> float:
    gap = max(0.0, train_summary.macro_f1 - val_summary.macro_f1)
    return float(val_summary.macro_f1 - 0.75 * gap)


def build_subtype_model(seed: int, n_jobs: int, num_classes: int, device_params: dict | None = None) -> LGBMClassifier:
    device_params = device_params or {}
    common_kwargs = dict(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=6,
        min_child_samples=120,
        subsample=0.75,
        colsample_bytree=0.70,
        reg_alpha=0.8,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight=None,
        verbose=-1,
        **device_params,
    )
    if num_classes == 2:
        return LGBMClassifier(
            objective="binary",
            **common_kwargs,
        )
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        **common_kwargs,
    )


def build_single_eval_fit_kwargs() -> dict:
    """Return LightGBM fit kwargs without an eval set for tiny subtype splits."""
    return {"eval_metric": "multi_logloss"}


def hierarchical_predict(
    *,
    bundle: dict,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    family_model: LGBMClassifier = bundle["family_model"]
    subtype_models: dict[str, LGBMClassifier] = bundle["subtype_models"]
    passthrough_labels: dict[str, str] = bundle["passthrough_labels"]

    family_prob = family_model.predict_proba(X)
    family_classes = family_model.classes_
    top_k_families = int(bundle.get("top_k_families", 1))

    family_pred = np.empty(len(X), dtype=object)
    family_conf = np.zeros(len(X), dtype="float64")
    subtype_pred = np.empty(len(X), dtype=object)
    subtype_conf = np.zeros(len(X), dtype="float64")

    for i in range(len(X)):
        family_order = np.argsort(family_prob[i])[::-1][:top_k_families]
        best_family = None
        best_family_prob = -1.0
        best_subtype = None
        best_subtype_prob = -1.0
        best_combined = -1.0

        for family_idx in family_order:
            family_name = str(family_classes[family_idx])
            family_probability = float(family_prob[i, family_idx])

            if family_name in passthrough_labels:
                subtype_name = passthrough_labels[family_name]
                subtype_probability = 1.0
            else:
                model = subtype_models[family_name]
                probs = model.predict_proba(X.iloc[[i]])[0]
                pred_idx = int(np.argmax(probs))
                subtype_name = str(model.classes_[pred_idx])
                subtype_probability = float(probs[pred_idx])

            combined_score = family_probability * subtype_probability
            if combined_score > best_combined:
                best_combined = combined_score
                best_family = family_name
                best_family_prob = family_probability
                best_subtype = subtype_name
                best_subtype_prob = subtype_probability

        family_pred[i] = best_family
        family_conf[i] = best_family_prob
        subtype_pred[i] = best_subtype
        subtype_conf[i] = best_subtype_prob

    return family_pred, family_conf, subtype_pred, subtype_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical family -> subtype LightGBM models.")
    parser.add_argument("--archive-dir", type=Path, default=Path("data/archive (2)"), help="Archive CSV folder.")
    parser.add_argument("--max-per-label", type=int, default=20_000, help="Balanced archive cap per subtype label.")
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=80_000,
        help="Cap used only for Stage 1 family training. Keeps family distribution more natural than full subtype balancing.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers.")
    parser.add_argument(
        "--top-k-families",
        type=int,
        default=2,
        help="During hierarchical prediction, score this many top family candidates and choose the best family+subtype combination.",
    )
    parser.add_argument("--deoverlap-official-test", action="store_true", help="Remove archive rows overlapping the official test set.")
    parser.add_argument("--skip-reports", action="store_true", help="Skip long classification reports.")
    add_lightgbm_device_args(parser)
    args = parser.parse_args()
    lgbm_device_params = lightgbm_device_params(
        args.lgbm_device,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
    )
    print_lightgbm_device(args.lgbm_device)

    t0 = time.time()
    print("Loading official test set.")
    test_df, X_test, y_test = load_official_test()
    test_df = add_family_column(test_df, label_col=TARGET_COL, family_col="family")
    y_test_family = test_df["family"]

    print("\nSampling balanced training data from archive.")
    df_sample = sample_archive_balanced(
        archive_dir=args.archive_dir,
        max_per_label=args.max_per_label,
        seed=float(args.seed) / 100.0,
    )
    if args.deoverlap_official_test:
        df_sample, removed = exclude_overlap_rows(
            df_train_pool=df_sample,
            X_reference=X_test,
            label="OFFICIAL TEST",
        )
        print("Rows removed during de-overlap:", removed)

    df_sample = add_family_column(df_sample, label_col=TARGET_COL, family_col="family")
    df_train, df_val = split_train_val(df_sample, val_frac=args.val_frac, seed=args.seed)
    df_family_pool = sample_family_capped(df_sample, family_col="family", max_per_family=args.max_per_family, seed=args.seed)
    df_family_train, df_family_val = split_train_val(df_family_pool, val_frac=args.val_frac, seed=args.seed)

    X_train = df_train[EXPECTED_COLUMNS].astype("float32")
    y_train_subtype = df_train[TARGET_COL]
    X_val = df_val[EXPECTED_COLUMNS].astype("float32")
    y_val_subtype = df_val[TARGET_COL]
    X_test = X_test.astype("float32")

    X_family_train = df_family_train[EXPECTED_COLUMNS].astype("float32")
    y_train_family = df_family_train["family"]
    X_family_val = df_family_val[EXPECTED_COLUMNS].astype("float32")
    y_val_family = df_family_val["family"]

    print("Subtype train/val shapes:", X_train.shape, X_val.shape)
    print("Family train/val shapes :", X_family_train.shape, X_family_val.shape)
    print("Families:", families_in_order())
    print("\nFamily pool counts:")
    print(df_family_pool["family"].value_counts().sort_index().to_string())

    family_weights = compute_family_class_weights(y_train_family)
    family_fit_kwargs = build_lgbm_fit_kwargs(X_family_val, y_val_family)

    print("\nTuning family model candidates...")
    family_candidates: list[tuple[str, ScoreSummary, ScoreSummary, LGBMClassifier]] = []
    for candidate_name, candidate_params in build_family_candidates(
        args.seed,
        args.n_jobs,
        family_weights,
        num_classes=int(y_train_family.nunique()),
        device_params=lgbm_device_params,
    ):
        print(f"- training {candidate_name}")
        candidate_model = LGBMClassifier(**candidate_params)
        candidate_model.fit(X_family_train, y_train_family, **family_fit_kwargs)
        candidate_train = summarize(y_train_family, candidate_model.predict(X_family_train))
        candidate_val = summarize(y_val_family, candidate_model.predict(X_family_val))
        family_candidates.append((candidate_name, candidate_train, candidate_val, candidate_model))

    family_candidates.sort(key=lambda item: -family_selection_score(item[1], item[2]))
    print("\n=== FAMILY TUNING SUMMARY ===")
    print(f"{'Candidate':<26} {'Train MF1':>10} {'Val MF1':>10} {'Gap':>10} {'Val Score':>10}")
    print("-" * 76)
    for candidate_name, train_score, val_score, _ in family_candidates:
        gap = train_score.macro_f1 - val_score.macro_f1
        print(
            f"{candidate_name:<26} "
            f"{train_score.macro_f1:>10.4f} {val_score.macro_f1:>10.4f} "
            f"{gap:>10.4f} {family_selection_score(train_score, val_score):>10.4f}"
        )

    best_family_name, family_train_summary, family_val_summary, family_model = family_candidates[0]
    print("\nSelected family model:", best_family_name)
    print("Selection rule: highest validation score = Val Macro-F1 - 0.75 * Train-Val gap")

    family_test_pred = family_model.predict(X_test)
    print_summary("FAMILY TRAIN", family_train_summary)
    print_summary("FAMILY VAL", family_val_summary)
    print_summary("FAMILY OFFICIAL TEST", summarize(y_test_family, family_test_pred))

    subtype_models: dict[str, LGBMClassifier] = {}
    passthrough_labels: dict[str, str] = {}

    print("\nTraining subtype models per family...")
    for family in families_in_order():
        subtype_labels = labels_for_family(family)
        if len(subtype_labels) == 1:
            passthrough_labels[family] = subtype_labels[0]
            print(f"- {family}: passthrough -> {subtype_labels[0]}")
            continue

        subtype_train_mask = (df_train["family"] == family).to_numpy()
        subtype_val_mask = (df_val["family"] == family).to_numpy()
        X_subtype_train = X_train.iloc[subtype_train_mask]
        y_subtype_train = y_train_subtype.iloc[subtype_train_mask]
        X_subtype_val = X_val.iloc[subtype_val_mask]
        y_subtype_val = y_val_subtype.iloc[subtype_val_mask]

        if y_subtype_train.nunique() <= 1:
            passthrough_labels[family] = str(y_subtype_train.iloc[0])
            print(f"- {family}: only one observed subtype in train -> passthrough")
            continue

        subtype_model = build_subtype_model(
            args.seed,
            args.n_jobs,
            num_classes=int(y_subtype_train.nunique()),
            device_params=lgbm_device_params,
        )
        if len(X_subtype_val) == 0 or y_subtype_val.nunique() <= 1:
            subtype_fit_kwargs = build_single_eval_fit_kwargs()
            print(f"- {family}: validation slice has <=1 subtype class, training without eval_set")
        else:
            subtype_fit_kwargs = build_lgbm_fit_kwargs(X_subtype_val, y_subtype_val)
        subtype_model.fit(X_subtype_train, y_subtype_train, **subtype_fit_kwargs)
        subtype_models[family] = subtype_model
        print(f"- {family}: trained {y_subtype_train.nunique()} subtype classes on {len(X_subtype_train):,} rows")

    bundle = {
        "model_type": "hierarchical",
        "feature_columns": EXPECTED_COLUMNS,
        "family_by_label": FAMILY_BY_LABEL,
        "family_model": family_model,
        "family_model_name": best_family_name,
        "subtype_models": subtype_models,
        "passthrough_labels": passthrough_labels,
        "families": families_in_order(),
        "top_k_families": int(args.top_k_families),
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print("\nSaved hierarchical bundle to:", BUNDLE_PATH)

    family_pred_val, family_conf_val, subtype_pred_val, subtype_conf_val = hierarchical_predict(bundle=bundle, X=X_val)
    family_pred_test, family_conf_test, subtype_pred_test, subtype_conf_test = hierarchical_predict(bundle=bundle, X=X_test)

    print_summary("END-TO-END SUBTYPE VAL", summarize(y_val_subtype, subtype_pred_val))
    print_summary("END-TO-END SUBTYPE OFFICIAL TEST", summarize(y_test, subtype_pred_test))

    if not args.skip_reports:
        print("\nClassification report (family official test):")
        print(classification_report(y_test_family, family_pred_test, zero_division=0))
        print("\nClassification report (subtype official test):")
        print(classification_report(y_test, subtype_pred_test, zero_division=0))

    correct_family_mask = (family_pred_test == y_test_family.to_numpy())
    if correct_family_mask.any():
        oracle_subtype_summary = summarize(
            y_test[correct_family_mask],
            subtype_pred_test[correct_family_mask],
        )
        print_summary("SUBTYPE OFFICIAL TEST (when family correct)", oracle_subtype_summary)
        print("Family confidence mean:", round(float(np.mean(family_conf_test)), 6))
        print("Subtype confidence mean:", round(float(np.mean(subtype_conf_test)), 6))

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
