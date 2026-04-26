from pathlib import Path
from functools import lru_cache
import json

import joblib
import numpy as np
import pandas as pd

MODEL_PATHS = {
    "Improved GPU LightGBM": Path("models/archive_lightgbm.joblib"),
    "Logistic Regression": Path("models/archive_logreg.joblib"),
    "Random Forest": Path("models/archive_random_forest.joblib"),
    "Hierarchical Family + Subtype": Path("models/hierarchical_lightgbm_bundle.joblib"),
}

DEFAULT_MODEL_NAME = "Hierarchical Family + Subtype"

THRESHOLD_CONFIG_PATHS = {
    # Threshold configs are only used by flat probabilistic models.
    "Logistic Regression": Path("models/archive_logreg_thresholds.json"),
    "Random Forest": Path("models/archive_random_forest_thresholds.json"),
}

DROP_COLS = [
    "ece_flag_number",
    "cwr_flag_number",
    "Telnet",
    "SMTP",
    "IRC",
    "DHCP",
]

TARGET_COL = "label"

EXPECTED_COLUMNS = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "SSH",
    "TCP",
    "UDP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]


def available_models() -> list[str]:
    return list(MODEL_PATHS.keys())


@lru_cache(maxsize=None)
def load_model(model_name: str = DEFAULT_MODEL_NAME):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available: {available_models()}"
        )

    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


@lru_cache(maxsize=None)
def load_threshold_config(model_name: str = DEFAULT_MODEL_NAME) -> dict:
    path = THRESHOLD_CONFIG_PATHS.get(model_name)
    if path is None or not path.exists():
        return {
            "enabled": False,
            "thresholds": {},
            "rare_classes": [],
            "fallback_to_next_rank": True,
            "abstain_on_low_confidence_rare": False,
            "abstain_label": "UNCERTAIN_RARE",
            "uncertainty_floor": 0.50,
        }

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("enabled", True)
    config.setdefault("thresholds", {})
    config.setdefault("rare_classes", list(config["thresholds"].keys()))
    config.setdefault("fallback_to_next_rank", True)
    config.setdefault("abstain_on_low_confidence_rare", False)
    config.setdefault("abstain_label", "UNCERTAIN_RARE")
    config.setdefault("uncertainty_floor", 0.50)
    return config


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors="ignore")

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL], errors="ignore")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df[EXPECTED_COLUMNS].copy()


def get_alert_level(prediction: str, confidence: float) -> str:
    if prediction == "BenignTraffic":
        if confidence >= 0.90:
            return "INFO"
        return "REVIEW"

    if confidence >= 0.90:
        return "HIGH_ALERT"
    if confidence >= 0.60:
        return "MEDIUM_ALERT"
    return "LOW_ALERT_REVIEW"


def is_uncertain(confidence: float) -> bool:
    return confidence < 0.50


def choose_thresholded_prediction(
    probs: np.ndarray,
    class_names: np.ndarray,
    threshold_config: dict,
) -> dict:
    ranked_indices = probs.argsort()[::-1]
    ranked = []
    for rank, idx in enumerate(ranked_indices[:3], start=1):
        ranked.append(
            {
                "rank": rank,
                "class_name": str(class_names[idx]),
                "probability": float(probs[idx]),
            }
        )

    top_choice = ranked[0]
    thresholds = threshold_config.get("thresholds", {})
    fallback_enabled = bool(threshold_config.get("fallback_to_next_rank", True))
    abstain_enabled = bool(threshold_config.get("abstain_on_low_confidence_rare", False))
    abstain_label = str(threshold_config.get("abstain_label", "UNCERTAIN_RARE"))
    rare_classes = set(threshold_config.get("rare_classes", []))

    selected = top_choice
    threshold_applied = False
    fallback_from_class = None
    abstained = False
    abstain_from_class = None

    if threshold_config.get("enabled", False):
        top_threshold = thresholds.get(top_choice["class_name"])
        top_is_low = top_threshold is not None and top_choice["probability"] < float(top_threshold)

        if abstain_enabled and top_choice["class_name"] in rare_classes and top_is_low:
            threshold_applied = True
            abstained = True
            abstain_from_class = top_choice["class_name"]
            selected = {
                "rank": 1,
                "class_name": abstain_label,
                "probability": top_choice["probability"],
            }
        else:
            for candidate in ranked:
                candidate_threshold = thresholds.get(candidate["class_name"])
                if candidate_threshold is None or candidate["probability"] >= float(candidate_threshold):
                    selected = candidate
                    break
            if selected["rank"] != 1:
                threshold_applied = True
                fallback_from_class = top_choice["class_name"]
            elif top_is_low:
                threshold_applied = True
                fallback_from_class = top_choice["class_name"]
                if not fallback_enabled:
                    selected = top_choice

    uncertainty_floor = float(threshold_config.get("uncertainty_floor", 0.50))
    uncertain = selected["probability"] < uncertainty_floor or threshold_applied

    return {
        "predicted_class": selected["class_name"],
        "confidence": selected["probability"],
        "threshold_applied": threshold_applied,
        "fallback_from_class": fallback_from_class,
        "abstained": abstained,
        "abstain_from_class": abstain_from_class,
        "selected_rank": int(selected["rank"]),
        "uncertain": uncertain,
        "top_ranked": ranked,
    }


def predict_hierarchical_dataframe(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    X = preprocess_input(df)
    family_model = model_bundle["family_model"]
    subtype_models = model_bundle["subtype_models"]
    passthrough_labels = model_bundle.get("passthrough_labels", {})
    top_k_families = int(model_bundle.get("top_k_families", 1))

    family_prob = family_model.predict_proba(X)
    family_classes = family_model.classes_

    family_order_by_row = np.argsort(family_prob, axis=1)[:, ::-1][:, :top_k_families]
    effective_top_k = family_order_by_row.shape[1]
    best_family = np.empty(len(X), dtype=object)
    best_family_confidence = np.full(len(X), -1.0, dtype="float64")
    best_subtype_name = np.empty(len(X), dtype=object)
    best_subtype_confidence = np.full(len(X), -1.0, dtype="float64")
    best_combined = np.full(len(X), -1.0, dtype="float64")
    best_top_subtypes: list[list[dict] | None] = [None] * len(X)

    for rank_idx in range(effective_top_k):
        row_indices_by_family: dict[str, list[int]] = {}
        family_indices_for_rank = family_order_by_row[:, rank_idx]
        for row_idx, family_idx in enumerate(family_indices_for_rank):
            family_name = str(family_classes[family_idx])
            row_indices_by_family.setdefault(family_name, []).append(row_idx)

        for family_name, row_indices in row_indices_by_family.items():
            row_array = np.asarray(row_indices, dtype=int)
            family_class_idx = int(np.where(family_classes == family_name)[0][0])
            family_conf = family_prob[row_array, family_class_idx]

            if family_name in passthrough_labels:
                subtype_name = passthrough_labels[family_name]
                subtype_conf = np.ones(len(row_array), dtype="float64")
                top_subtypes = [
                    [
                        {"class_name": subtype_name, "probability": 1.0},
                        {"class_name": subtype_name, "probability": 0.0},
                        {"class_name": subtype_name, "probability": 0.0},
                    ]
                    for _ in row_array
                ]
            else:
                subtype_model = subtype_models[family_name]
                subtype_prob = subtype_model.predict_proba(X.iloc[row_array])
                subtype_classes = subtype_model.classes_
                subtype_order = np.argsort(subtype_prob, axis=1)[:, ::-1]
                subtype_top_idx = subtype_order[:, 0]
                subtype_names = subtype_classes[subtype_top_idx].astype(str)
                subtype_conf = subtype_prob[np.arange(len(row_array)), subtype_top_idx]
                top_subtypes = []
                for local_idx, ordered_indices in enumerate(subtype_order[:, :3]):
                    ranked = [
                        {
                            "class_name": str(subtype_classes[idx]),
                            "probability": float(subtype_prob[local_idx, idx]),
                        }
                        for idx in ordered_indices
                    ]
                    while len(ranked) < 3:
                        ranked.append({"class_name": ranked[0]["class_name"], "probability": 0.0})
                    top_subtypes.append(ranked)

            combined_scores = family_conf * subtype_conf
            improve_mask = combined_scores > best_combined[row_array]
            improved_rows = row_array[improve_mask]
            if len(improved_rows) == 0:
                continue

            best_combined[improved_rows] = combined_scores[improve_mask]
            best_family[improved_rows] = family_name
            best_family_confidence[improved_rows] = family_conf[improve_mask]

            improved_local_indices = np.flatnonzero(improve_mask)
            if family_name in passthrough_labels:
                for row_idx, local_idx in zip(improved_rows, improved_local_indices):
                    best_subtype_name[row_idx] = subtype_name
                    best_subtype_confidence[row_idx] = subtype_conf[local_idx]
                    best_top_subtypes[row_idx] = top_subtypes[local_idx]
            else:
                for row_idx, local_idx in zip(improved_rows, improved_local_indices):
                    best_subtype_name[row_idx] = subtype_names[local_idx]
                    best_subtype_confidence[row_idx] = subtype_conf[local_idx]
                    best_top_subtypes[row_idx] = top_subtypes[local_idx]

    results = []
    for i in range(len(X)):
        family_name = str(best_family[i])
        family_confidence = float(best_family_confidence[i])
        subtype_name = str(best_subtype_name[i])
        subtype_confidence = float(best_subtype_confidence[i])
        top_subtypes = best_top_subtypes[i]
        if top_subtypes is None:
            top_subtypes = [
                {"class_name": subtype_name, "probability": subtype_confidence},
                {"class_name": subtype_name, "probability": 0.0},
                {"class_name": subtype_name, "probability": 0.0},
            ]

        overall_confidence = float(family_confidence * subtype_confidence)
        results.append(
            {
                "predicted_family": family_name,
                "family_confidence": family_confidence,
                "predicted_class": subtype_name,
                "subtype_confidence": subtype_confidence,
                "confidence": overall_confidence,
                "alert_level": get_alert_level(subtype_name, overall_confidence),
                "uncertain": is_uncertain(overall_confidence),
                "threshold_applied": False,
                "fallback_from_class": None,
                "abstained": False,
                "abstain_from_class": None,
                "selected_rank": 1,
                "top_1_class": top_subtypes[0]["class_name"],
                "top_1_prob": top_subtypes[0]["probability"],
                "top_2_class": top_subtypes[1]["class_name"],
                "top_2_prob": top_subtypes[1]["probability"],
                "top_3_class": top_subtypes[2]["class_name"],
                "top_3_prob": top_subtypes[2]["probability"],
            }
        )

    return pd.DataFrame(results)


def predict_dataframe(df: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME) -> pd.DataFrame:
    model = load_model(model_name=model_name)
    if isinstance(model, dict) and model.get("model_type") == "hierarchical":
        return predict_hierarchical_dataframe(df, model)

    threshold_config = load_threshold_config(model_name=model_name)
    X = preprocess_input(df)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")

    probabilities = model.predict_proba(X)
    class_names = model.classes_

    results = []

    for i in range(len(X)):
        probs = probabilities[i]
        decision = choose_thresholded_prediction(probs, class_names, threshold_config)
        top_ranked = decision["top_ranked"]

        top_1_class = top_ranked[0]["class_name"]
        top_1_prob = top_ranked[0]["probability"]
        top_2_class = top_ranked[1]["class_name"]
        top_2_prob = top_ranked[1]["probability"]
        top_3_class = top_ranked[2]["class_name"]
        top_3_prob = top_ranked[2]["probability"]

        results.append({
            "predicted_class": decision["predicted_class"],
            "confidence": decision["confidence"],
            "alert_level": "REVIEW" if decision["abstained"] else get_alert_level(decision["predicted_class"], decision["confidence"]),
            "uncertain": bool(decision["uncertain"]),
            "threshold_applied": bool(decision["threshold_applied"]),
            "fallback_from_class": decision["fallback_from_class"],
            "abstained": bool(decision["abstained"]),
            "abstain_from_class": decision["abstain_from_class"],
            "selected_rank": int(decision["selected_rank"]),
            "top_1_class": top_1_class,
            "top_1_prob": top_1_prob,
            "top_2_class": top_2_class,
            "top_2_prob": top_2_prob,
            "top_3_class": top_3_class,
            "top_3_prob": top_3_prob,
        })

    return pd.DataFrame(results)
