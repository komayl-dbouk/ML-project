from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from lightgbm_device import add_lightgbm_device_args, lightgbm_device_params, print_lightgbm_device

try:
    from lightgbm import early_stopping, log_evaluation
except ImportError:  # pragma: no cover - compatibility fallback
    early_stopping = None
    log_evaluation = None


# Keep these in sync with inference/eval
from inference_engine import EXPECTED_COLUMNS, DROP_COLS, TARGET_COL  # noqa: E402


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_TEST_CSV = Path("data/raw/CICIoT2023/CICIOT23/test/test.csv")


@dataclass(frozen=True)
class EvalSummary:
    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    rare_mean_recall: float
    rare_mean_precision: float
    rare_mean_auprc: float
    rare_classes: list[str]


def _quote_ident(col: str) -> str:
    # DuckDB uses double quotes for identifiers
    return '"' + col.replace('"', '""') + '"'


def load_official_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not OFFICIAL_TEST_CSV.exists():
        raise FileNotFoundError(f"Official test CSV not found: {OFFICIAL_TEST_CSV}")

    df = pd.read_csv(OFFICIAL_TEST_CSV, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column not found in official test CSV.")

    # Drop training-time drop cols if present
    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors="ignore")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Official test is missing required columns: {missing}")

    X = df[EXPECTED_COLUMNS].copy()
    y = df[TARGET_COL].copy()
    return df, X, y


def sample_archive_balanced(
    archive_dir: Path,
    max_per_label: int,
    seed: float,
) -> pd.DataFrame:
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive folder not found: {archive_dir}")

    # DuckDB can glob directly; forward slashes are fine on Windows too.
    glob_path = (archive_dir / "*.csv").as_posix()

    wanted_cols = list(EXPECTED_COLUMNS) + [TARGET_COL]
    select_cols_sql = ", ".join(_quote_ident(c) for c in wanted_cols)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=false;")
    con.execute(f"SELECT setseed({seed});")

    # union_by_name=true makes it robust if column order differs
    base_sql = f"""
        SELECT {select_cols_sql}
        FROM read_csv_auto('{glob_path}', union_by_name=true, all_varchar=false)
        WHERE {_quote_ident(TARGET_COL)} IS NOT NULL
    """

    # Balanced sample: cap each label to max_per_label using a random ordering
    sample_sql = f"""
        WITH base AS (
            {base_sql}
        )
        SELECT *
        FROM base
        QUALIFY row_number() OVER (
            PARTITION BY {_quote_ident(TARGET_COL)}
            ORDER BY random()
        ) <= {int(max_per_label)}
    """

    df = con.execute(sample_sql).df()
    con.close()

    # Ensure exact order: features then label
    df = df[wanted_cols].copy()
    return df


def split_train_val(
    df: pd.DataFrame,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1.0 - val_frac))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def eval_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rare_support_threshold: int,
    *,
    dataset_name: str = "OFFICIAL TEST",
    print_report: bool = True,
) -> EvalSummary:
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro = float(f1_score(y_test, y_pred, average="macro"))
    weighted = float(f1_score(y_test, y_pred, average="weighted"))
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Rare-class recall is most meaningful on the official test distribution.
    if dataset_name.upper() == "OFFICIAL TEST":
        labels, support = np.unique(y_test, return_counts=True)
        rare_mask = support <= rare_support_threshold
        rare_labels = labels[rare_mask].tolist()

        if rare_labels:
            # Compute rare-class precision/recall on official-test labels.
            p, r, _, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                labels=rare_labels,
                zero_division=0,
            )
            rare_mean_precision = float(np.mean(p)) if len(p) else 0.0
            rare_mean_recall = float(np.mean(r)) if len(r) else 0.0
            if y_prob is not None:
                classes = np.asarray(model.classes_)
                y_true_bin = label_binarize(y_test, classes=classes)
                ap_values = []
                for rare_label in rare_labels:
                    label_idx = int(np.where(classes == rare_label)[0][0])
                    ap_values.append(float(average_precision_score(y_true_bin[:, label_idx], y_prob[:, label_idx])))
                rare_mean_auprc = float(np.mean(ap_values)) if ap_values else float("nan")
            else:
                rare_mean_auprc = float("nan")
        else:
            rare_mean_precision = float("nan")
            rare_mean_recall = float("nan")
            rare_mean_auprc = float("nan")
    else:
        rare_mean_precision = float("nan")
        rare_mean_recall = float("nan")
        rare_mean_auprc = float("nan")
        rare_labels = []

    print(f"\n=== {name} ({dataset_name}) ===")
    print("Accuracy    :", round(acc, 6))
    print("Macro-F1    :", round(macro, 6))
    print("Weighted-F1 :", round(weighted, 6))
    if dataset_name.upper() == "OFFICIAL TEST" and rare_labels:
        print(f"Rare mean precision (support <= {rare_support_threshold}):", round(rare_mean_precision, 6))
        print(f"Rare mean recall (support <= {rare_support_threshold}):", round(rare_mean_recall, 6))
        if not np.isnan(rare_mean_auprc):
            print(f"Rare mean AUPRC (OvR, support <= {rare_support_threshold}):", round(rare_mean_auprc, 6))
        print("Rare labels:", rare_labels)
    elif dataset_name.upper() == "OFFICIAL TEST":
        print(f"Rare mean precision (support <= {rare_support_threshold}): N/A (no rare classes on this test)")
        print(f"Rare mean recall (support <= {rare_support_threshold}): N/A (no rare classes on this test)")

    if print_report:
        print(f"\nClassification report ({dataset_name.lower()}):")
        print(classification_report(y_test, y_pred, zero_division=0))

    return EvalSummary(
        name=name,
        accuracy=acc,
        macro_f1=macro,
        weighted_f1=weighted,
        rare_mean_recall=rare_mean_recall,
        rare_mean_precision=rare_mean_precision,
        rare_mean_auprc=rare_mean_auprc,
        rare_classes=rare_labels,
    )


def build_lgbm_fit_kwargs(X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "multi_logloss",
    }
    callbacks = []
    if early_stopping is not None:
        callbacks.append(early_stopping(50, verbose=False))
    if log_evaluation is not None:
        callbacks.append(log_evaluation(period=0))
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    return fit_kwargs


def build_lgbm_candidates(random_seed: int, n_jobs: int, device_params: dict | None = None) -> list[tuple[str, dict]]:
    device_params = device_params or {}
    return [
        (
            "lgbm_balanced_baseline",
            dict(
                objective="multiclass",
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=23,
                max_depth=8,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=1.5,
                random_state=random_seed,
                n_jobs=n_jobs,
                class_weight=None,
                verbose=-1,
                **device_params,
            ),
        ),
        (
            "lgbm_stable_medium",
            dict(
                objective="multiclass",
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=6,
                min_child_samples=140,
                subsample=0.75,
                colsample_bytree=0.70,
                reg_alpha=0.8,
                reg_lambda=3.0,
                random_state=random_seed,
                n_jobs=n_jobs,
                class_weight=None,
                verbose=-1,
                **device_params,
            ),
        ),
        (
            "lgbm_stable_strong",
            dict(
                objective="multiclass",
                n_estimators=1400,
                learning_rate=0.025,
                num_leaves=11,
                max_depth=5,
                min_child_samples=200,
                subsample=0.70,
                colsample_bytree=0.70,
                reg_alpha=1.2,
                reg_lambda=4.0,
                random_state=random_seed,
                n_jobs=n_jobs,
                class_weight=None,
                min_split_gain=0.05,
                max_bin=127,
                verbose=-1,
                **device_params,
            ),
        ),
        (
            "lgbm_stable_ultra",
            dict(
                objective="multiclass",
                n_estimators=1600,
                learning_rate=0.02,
                num_leaves=9,
                max_depth=4,
                min_child_samples=260,
                subsample=0.65,
                colsample_bytree=0.65,
                reg_alpha=1.5,
                reg_lambda=5.0,
                min_split_gain=0.10,
                max_bin=127,
                random_state=random_seed,
                n_jobs=n_jobs,
                class_weight=None,
                verbose=-1,
                **device_params,
            ),
        ),
    ]


def overfit_gap(train_summary: EvalSummary, val_summary: EvalSummary) -> float:
    return float(train_summary.macro_f1 - val_summary.macro_f1)


def validation_selection_score(train_summary: EvalSummary, val_summary: EvalSummary) -> float:
    gap = max(0.0, overfit_gap(train_summary, val_summary))
    return float(val_summary.macro_f1 - 0.75 * gap)


def print_lgbm_tuning_table(candidates: list[tuple[str, EvalSummary, EvalSummary]]) -> None:
    print("\n=== LIGHTGBM TUNING SUMMARY ===")
    print(
        f"{'Candidate':<26} {'Train MF1':>10} {'Val MF1':>10} {'Gap':>10} {'Val Score':>10}"
    )
    print("-" * 76)
    for candidate_name, train_summary, val_summary in candidates:
        print(
            f"{candidate_name:<26} "
            f"{train_summary.macro_f1:>10.4f} {val_summary.macro_f1:>10.4f} "
            f"{overfit_gap(train_summary, val_summary):>10.4f} "
            f"{validation_selection_score(train_summary, val_summary):>10.4f}"
        )


def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    # Stable, vectorized row hash for duplicate/overlap checks.
    # uint64 hashes are enough for practical collision resistance here.
    return pd.util.hash_pandas_object(df, index=False).to_numpy(dtype="uint64", copy=False)


def report_overlap(
    *,
    name_a: str,
    X_a: pd.DataFrame,
    name_b: str,
    X_b: pd.DataFrame,
) -> None:
    ha = _row_hashes(X_a)
    hb = _row_hashes(X_b)
    ha_u = np.unique(ha)
    hb_u = np.unique(hb)
    overlap = np.intersect1d(ha_u, hb_u, assume_unique=True)
    overlap_n = int(overlap.size)
    print("\n=== DUPLICATE/OVERLAP CHECK (by row-hash of EXPECTED_COLUMNS) ===")
    print(f"{name_a} unique rows:", int(ha_u.size))
    print(f"{name_b} unique rows:", int(hb_u.size))
    print("Overlapping unique rows:", overlap_n)
    if ha_u.size:
        print(f"Overlap vs {name_a}: {overlap_n / ha_u.size:.6%}")
    if hb_u.size:
        print(f"Overlap vs {name_b}: {overlap_n / hb_u.size:.6%}")


def exclude_overlap_rows(
    *,
    df_train_pool: pd.DataFrame,
    X_reference: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, int]:
    """Drop any rows from the training pool whose feature hash appears in a reference set."""
    train_hash = _row_hashes(df_train_pool[EXPECTED_COLUMNS])
    ref_hash = np.unique(_row_hashes(X_reference))
    keep_mask = ~np.isin(train_hash, ref_hash, assume_unique=False)
    removed = int((~keep_mask).sum())

    print(f"\n=== DE-OVERLAP AGAINST {label} ===")
    print("Rows before filtering:", len(df_train_pool))
    print("Rows removed:", removed)
    print("Rows kept:", int(keep_mask.sum()))

    filtered = df_train_pool.loc[keep_mask].reset_index(drop=True)
    return filtered, removed


def print_summary_table(
    *,
    scores: dict[str, dict[str, EvalSummary]],
    overlap_before: int,
    overlap_after: int,
    removed_overlap_rows: int,
) -> None:
    print("\n=== MODEL SUMMARY ===")
    print(
        f"{'Model':<15} {'Train MF1':>10} {'Val MF1':>10} {'Test MF1':>10} "
        f"{'Train-Val':>10} {'Val-Test':>10} {'Rare Prec':>10} {'Rare Rec':>10} {'Rare AP':>10}"
    )
    print("-" * 80)

    for model_name, by_split in scores.items():
        train = by_split.get("TRAIN")
        val = by_split.get("VAL")
        test = by_split.get("OFFICIAL TEST")

        train_macro = train.macro_f1 if train else float("nan")
        val_macro = val.macro_f1 if val else float("nan")
        test_macro = test.macro_f1 if test else float("nan")
        train_val_gap = train_macro - val_macro if train and val else float("nan")
        val_test_gap = val_macro - test_macro if val and test else float("nan")
        rare_precision = test.rare_mean_precision if test else float("nan")
        rare_recall = test.rare_mean_recall if test else float("nan")
        rare_auprc = test.rare_mean_auprc if test else float("nan")

        print(
            f"{model_name:<15} "
            f"{train_macro:>10.4f} {val_macro:>10.4f} {test_macro:>10.4f} "
            f"{train_val_gap:>10.4f} {val_test_gap:>10.4f} {rare_precision:>10.4f} {rare_recall:>10.4f} {rare_auprc:>10.4f}"
        )

    print("\nData hygiene:")
    print("Overlap before filtering :", overlap_before)
    print("Overlap rows removed     :", removed_overlap_rows)
    print("Overlap after filtering  :", overlap_after)

    print("\nInterpretation guide:")
    print("- Large Train-Val gap usually indicates overfitting.")
    print("- Large Val-Test gap suggests validation is easier than the untouched test.")
    print("- Rare Prec tracks precision on rare official-test classes only.")
    print("- Rare Rec tracks recall on rare official-test classes only.")
    print("- Rare AP is one-vs-rest AUPRC averaged over rare official-test classes.")


def main():
    parser = argparse.ArgumentParser(description="Train 3 models on archive dataset and evaluate on official test.")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/archive (2)"),
        help="Folder containing many CSVs (12GB archive).",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=20_000,
        help="Balanced sample cap per label from archive (controls RAM/time).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction (from the sampled archive dataset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for tree models. Use 1 if your environment restricts multiprocessing/thread pools.",
    )
    parser.add_argument(
        "--rare-support-threshold",
        type=int,
        default=500,
        help="Defines 'rare' by support in the official test set.",
    )
    parser.add_argument(
        "--sanity-checks",
        action="store_true",
        help="Run leakage/overfitting checks: train/val evaluation, shuffled-label test, and overlap check.",
    )
    parser.add_argument(
        "--deoverlap-official-test",
        action="store_true",
        help="Remove any sampled archive rows whose features also appear in the official test set.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip printing full classification_report to keep output short (metrics still printed).",
    )
    parser.add_argument(
        "--tune-lightgbm",
        action="store_true",
        help="Run a small LightGBM candidate sweep and pick the best candidate using validation Macro-F1 with an overfitting penalty.",
    )
    parser.add_argument(
        "--lgbm-preset",
        choices=[
            "default",
            "lgbm_balanced_baseline",
            "lgbm_stable_medium",
            "lgbm_stable_strong",
            "lgbm_stable_ultra",
        ],
        default="default",
        help="Train a single LightGBM preset when --tune-lightgbm is not used. Stable presets are more regularized and usually overfit less.",
    )
    parser.add_argument(
        "--skip-cpu-models",
        action="store_true",
        help="Train/evaluate only LightGBM. scikit-learn Logistic Regression and Random Forest are CPU-only in this project.",
    )
    add_lightgbm_device_args(parser)
    args = parser.parse_args()
    lgbm_device_params = lightgbm_device_params(
        args.lgbm_device,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
    )
    print_lightgbm_device(args.lgbm_device)

    t0 = time.time()
    print("Loading official test set (kept untouched).")
    _, X_official_test, y_official_test = load_official_test()

    print("\nSampling balanced training data from archive (streamed via DuckDB).")
    df_sample = sample_archive_balanced(
        archive_dir=args.archive_dir,
        max_per_label=args.max_per_label,
        seed=float(args.seed) / 100.0,
    )
    print("Sampled shape:", df_sample.shape)
    print("Labels in sample:", df_sample[TARGET_COL].nunique())

    overlap_before = 0
    overlap_after = 0
    removed_overlap_rows = 0

    if args.sanity_checks or args.deoverlap_official_test:
        sample_hashes = np.unique(_row_hashes(df_sample[EXPECTED_COLUMNS]))
        test_hashes = np.unique(_row_hashes(X_official_test))
        overlap_before = int(np.intersect1d(sample_hashes, test_hashes, assume_unique=True).size)

    if args.sanity_checks:
        report_overlap(name_a="ARCHIVE SAMPLE (train+val)", X_a=df_sample[EXPECTED_COLUMNS], name_b="OFFICIAL TEST", X_b=X_official_test)

    if args.deoverlap_official_test:
        df_sample, removed_overlap_rows = exclude_overlap_rows(
            df_train_pool=df_sample,
            X_reference=X_official_test,
            label="OFFICIAL TEST",
        )
        sample_hashes = np.unique(_row_hashes(df_sample[EXPECTED_COLUMNS]))
        test_hashes = np.unique(_row_hashes(X_official_test))
        overlap_after = int(np.intersect1d(sample_hashes, test_hashes, assume_unique=True).size)
        print("Remaining overlap after filtering:", overlap_after)

    df_train, df_val = split_train_val(df_sample, val_frac=args.val_frac, seed=args.seed)

    X_train = df_train[EXPECTED_COLUMNS].astype("float32")
    y_train = df_train[TARGET_COL]
    X_val = df_val[EXPECTED_COLUMNS].astype("float32")
    y_val = df_val[TARGET_COL]

    print("\nTrain/val shapes:", X_train.shape, X_val.shape)
    print_report = not args.skip_reports
    scores: dict[str, dict[str, EvalSummary]] = {}

    # -------------------------
    # 1) Logistic Regression
    # -------------------------
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    max_iter=300,
                    class_weight="balanced",
                    random_state=args.seed,
                    verbose=0,
                ),
            ),
        ]
    )

    if not args.skip_cpu_models:
        print("\nTraining Logistic Regression...")
        logreg.fit(X_train, y_train)
        joblib.dump(logreg, MODELS_DIR / "archive_logreg.joblib")
        print("Saved:", MODELS_DIR / "archive_logreg.joblib")
        if args.sanity_checks:
            scores.setdefault("LogReg", {})["TRAIN"] = eval_model(
                "LogReg", logreg, X_train, y_train, args.rare_support_threshold, dataset_name="TRAIN", print_report=False
            )
            scores.setdefault("LogReg", {})["VAL"] = eval_model(
                "LogReg", logreg, X_val, y_val, args.rare_support_threshold, dataset_name="VAL", print_report=print_report
            )

    # -------------------------
    # 2) Random Forest
    # -------------------------
    rf = RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=None,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=0,
    )
    if not args.skip_cpu_models:
        print("\nTraining Random Forest...")
        rf.fit(X_train, y_train)
        joblib.dump(rf, MODELS_DIR / "archive_random_forest.joblib")
        print("Saved:", MODELS_DIR / "archive_random_forest.joblib")
        if args.sanity_checks:
            scores.setdefault("RandomForest", {})["TRAIN"] = eval_model(
                "RandomForest", rf, X_train, y_train, args.rare_support_threshold, dataset_name="TRAIN", print_report=False
            )
            scores.setdefault("RandomForest", {})["VAL"] = eval_model(
                "RandomForest", rf, X_val, y_val, args.rare_support_threshold, dataset_name="VAL", print_report=print_report
            )

    # -------------------------
    # 3) LightGBM
    # -------------------------
    fit_kwargs = build_lgbm_fit_kwargs(X_val, y_val)

    if args.tune_lightgbm:
        print("\nTuning LightGBM candidates...")
        candidate_results: list[tuple[str, EvalSummary, EvalSummary, LGBMClassifier]] = []
        for candidate_name, candidate_params in build_lgbm_candidates(args.seed, args.n_jobs, lgbm_device_params):
            print(f"\nTraining {candidate_name}...")
            candidate_model = LGBMClassifier(**candidate_params)
            candidate_model.fit(X_train, y_train, **fit_kwargs)

            candidate_train = eval_model(
                candidate_name,
                candidate_model,
                X_train,
                y_train,
                args.rare_support_threshold,
                dataset_name="TRAIN",
                print_report=False,
            )
            candidate_val = eval_model(
                candidate_name,
                candidate_model,
                X_val,
                y_val,
                args.rare_support_threshold,
                dataset_name="VAL",
                print_report=False,
            )
            candidate_results.append((candidate_name, candidate_train, candidate_val, candidate_model))

        candidate_results.sort(key=lambda item: -validation_selection_score(item[1], item[2]))
        print_lgbm_tuning_table([(name, train_s, val_s) for name, train_s, val_s, _ in candidate_results])

        best_candidate_name, best_train_summary, best_val_summary, lgbm = candidate_results[0]
        print("\nSelected LightGBM candidate:", best_candidate_name)
        print("Selection rule: highest validation score = Val Macro-F1 - 0.75 * Train-Val gap")
        joblib.dump(lgbm, MODELS_DIR / "archive_lightgbm.joblib")
        print("Saved:", MODELS_DIR / "archive_lightgbm.joblib")
        if args.sanity_checks:
            scores.setdefault("LightGBM", {})["TRAIN"] = best_train_summary
            scores.setdefault("LightGBM", {})["VAL"] = best_val_summary
    else:
        print("\nTraining LightGBM...")
        if args.lgbm_preset == "default":
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
            lgbm_params = dict(
                objective="multiclass",
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=10,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                class_weight=class_weights,
                verbose=-1,
                **lgbm_device_params,
            )
        else:
            presets = dict(build_lgbm_candidates(args.seed, args.n_jobs, lgbm_device_params))
            lgbm_params = presets[args.lgbm_preset]
            print("Using preset:", args.lgbm_preset)
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train, **fit_kwargs)
        joblib.dump(lgbm, MODELS_DIR / "archive_lightgbm.joblib")
        print("Saved:", MODELS_DIR / "archive_lightgbm.joblib")
        if args.sanity_checks:
            scores.setdefault("LightGBM", {})["TRAIN"] = eval_model(
                "LightGBM", lgbm, X_train, y_train, args.rare_support_threshold, dataset_name="TRAIN", print_report=False
            )
            scores.setdefault("LightGBM", {})["VAL"] = eval_model(
                "LightGBM", lgbm, X_val, y_val, args.rare_support_threshold, dataset_name="VAL", print_report=print_report
            )

    # -------------------------
    # Evaluate on OFFICIAL TEST
    # -------------------------
    summaries: list[EvalSummary] = []
    if not args.skip_cpu_models:
        logreg_test = eval_model(
            "LogReg",
            logreg,
            X_official_test,
            y_official_test,
            args.rare_support_threshold,
            dataset_name="OFFICIAL TEST",
            print_report=print_report,
        )
        scores.setdefault("LogReg", {})["OFFICIAL TEST"] = logreg_test
        summaries.append(logreg_test)

        rf_test = eval_model(
            "RandomForest",
            rf,
            X_official_test,
            y_official_test,
            args.rare_support_threshold,
            dataset_name="OFFICIAL TEST",
            print_report=print_report,
        )
        scores.setdefault("RandomForest", {})["OFFICIAL TEST"] = rf_test
        summaries.append(rf_test)

    lgbm_test = eval_model(
        "LightGBM",
        lgbm,
        X_official_test,
        y_official_test,
        args.rare_support_threshold,
        dataset_name="OFFICIAL TEST",
        print_report=print_report,
    )
    scores.setdefault("LightGBM", {})["OFFICIAL TEST"] = lgbm_test
    summaries.append(lgbm_test)

    if args.sanity_checks:
        print("\n=== SHUFFLED-LABEL SANITY TEST (should perform near-random) ===")
        y_train_shuf = y_train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        shuffled_models = [("LightGBM", lgbm)]
        if not args.skip_cpu_models:
            shuffled_models = [("LogReg", logreg), ("RandomForest", rf)] + shuffled_models
        for model_name, base_model in shuffled_models:
            m = clone(base_model)
            print(f"\nTraining {model_name} on SHUFFLED labels...")
            if model_name == "LightGBM":
                shuffled_fit_kwargs = {
                    "eval_set": [(X_val, y_val)],
                    "eval_metric": "multi_logloss",
                }
                shuffled_callbacks = []
                if early_stopping is not None:
                    shuffled_callbacks.append(early_stopping(50, verbose=False))
                if log_evaluation is not None:
                    shuffled_callbacks.append(log_evaluation(period=0))
                if shuffled_callbacks:
                    shuffled_fit_kwargs["callbacks"] = shuffled_callbacks
                m.fit(X_train, y_train_shuf, **shuffled_fit_kwargs)
            else:
                m.fit(X_train, y_train_shuf)
            eval_model(
                f"{model_name} (shuffled)",
                m,
                X_val,
                y_val,
                args.rare_support_threshold,
                dataset_name="VAL",
                print_report=False,
            )
            eval_model(
                f"{model_name} (shuffled)",
                m,
                X_official_test,
                y_official_test,
                args.rare_support_threshold,
                dataset_name="OFFICIAL TEST",
                print_report=False,
            )

    if args.sanity_checks:
        validation_candidates: list[tuple[str, EvalSummary, EvalSummary]] = [
            ("LightGBM", scores["LightGBM"]["TRAIN"], scores["LightGBM"]["VAL"]),
        ]
        if not args.skip_cpu_models:
            validation_candidates = [
                ("LogReg", scores["LogReg"]["TRAIN"], scores["LogReg"]["VAL"]),
                ("RandomForest", scores["RandomForest"]["TRAIN"], scores["RandomForest"]["VAL"]),
            ] + validation_candidates
        validation_candidates.sort(key=lambda item: -validation_selection_score(item[1], item[2]))
        chosen_name, chosen_train, chosen_val = validation_candidates[0]
        print("\n=== MODEL CHOICE (validation only) ===")
        print("Chosen by: highest validation score = Val Macro-F1 - 0.75 * Train-Val gap")
        print("Best candidate:", chosen_name)
        print("Train Macro-F1:", round(chosen_train.macro_f1, 6))
        print("Val Macro-F1  :", round(chosen_val.macro_f1, 6))
        print("Train-Val gap :", round(overfit_gap(chosen_train, chosen_val), 6))
        print("Validation score:", round(validation_selection_score(chosen_train, chosen_val), 6))

    if args.sanity_checks:
        print_summary_table(
            scores=scores,
            overlap_before=overlap_before,
            overlap_after=overlap_after,
            removed_overlap_rows=removed_overlap_rows,
        )

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
