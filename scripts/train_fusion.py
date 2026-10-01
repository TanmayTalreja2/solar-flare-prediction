from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent.parent

FEATURES = (
    ROOT / "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)

LABELS = (
    ROOT / "data/processed/magnetograms/"
    "dataset_labels.csv"
)

XGB_PATH = (
    ROOT / "models/"
    "xgboost_2012_temporal_features.joblib"
)

CNN_PATH = (
    ROOT / "models/"
    "cnn_magnetogram.pt"
)

OUTPUT_DIR = ROOT / "results/ensemble"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Same chronological boundaries used to train XGBoost and the CNN.
# VALIDATION is used to fit + threshold the fusion model.
# TEST is used ONLY for the final, honest evaluation.
VALIDATION_START = pd.Timestamp("2012-05-01 00:00:00")
TEST_START = pd.Timestamp("2012-07-01 00:00:00")


# ============================================================
# CNN
# ============================================================

def load_cnn():

    import torch
    import torch.nn as nn
    import torchvision.models as models

    model = models.resnet18(
        weights=None
    )

    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 1)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.load_state_dict(
        torch.load(
            CNN_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


def get_cnn_predictions(labels, cnn_model=None, cnn_device=None):

    import torch

    if cnn_model is None:
        cnn_model, cnn_device = load_cnn()

    probabilities = []

    for _, row in labels.iterrows():

        fname = (
            f"harp_{int(row['HARPNUM'])}_"
            f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}_"
            f"t{int(row['target_24h'])}.npz"
        )

        path = (
            ROOT
            / "data/processed/magnetograms"
            / fname
        )

        if not path.exists():

            probabilities.append(
                np.nan
            )

            continue

        data = np.load(
            path
        )["img"]

        tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(cnn_device)

        with torch.no_grad():

            output = cnn_model(
                tensor
            ).view(-1)

            probability = torch.sigmoid(
                output
            ).item()

        probabilities.append(
            probability
        )

    return np.array(
        probabilities
    )


# ============================================================
# BUILD XGB + CNN PROBABILITIES FOR A GIVEN PERIOD
# ============================================================

def build_period_probabilities(
    data,
    labels,
    period_start,
    period_end,
    xgb_model,
    imputer,
    features,
    cnn_model,
    cnn_device,
    period_name,
):

    print(f"\nBuilding probabilities for {period_name} period...")

    if period_end is not None:
        period_data = data[
            (data["observation_time"] >= period_start)
            & (data["observation_time"] < period_end)
        ].copy()

        period_labels = labels[
            (labels["observation_time"] >= period_start)
            & (labels["observation_time"] < period_end)
        ].copy()

    else:
        period_data = data[
            data["observation_time"] >= period_start
        ].copy()

        period_labels = labels[
            labels["observation_time"] >= period_start
        ].copy()

    print(f"{period_name} feature rows : {len(period_data)}")
    print(f"{period_name} magnetogram rows: {len(period_labels)}")

    xgb_subset = period_data[
        [
            "HARPNUM",
            "NOAA_AR",
            "observation_time",
            "target_24h",
        ]
        + features
    ].copy()

    merged = period_labels.merge(
        xgb_subset,
        on=[
            "HARPNUM",
            "observation_time",
        ],
        how="inner",
        suffixes=("_label", "")
    )

    print(f"{period_name} matched observations: {len(merged)}")

    X = merged[features]
    X_imp = imputer.transform(X)
    merged["xgb_probability"] = xgb_model.predict_proba(X_imp)[:, 1]

    cnn_probs = get_cnn_predictions(
        merged[
            [
                "HARPNUM",
                "observation_time",
                "target_24h_label",
            ]
        ].rename(
            columns={"target_24h_label": "target_24h"}
        ),
        cnn_model=cnn_model,
        cnn_device=cnn_device,
    )

    merged["cnn_probability"] = cnn_probs

    merged = merged.dropna(
        subset=["xgb_probability", "cnn_probability"]
    ).reset_index(drop=True)

    print(f"{period_name} final matched samples: {len(merged)}")

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" XGBOOST + CNN FUSION (leakage-free)")
    print("=" * 60)

    print("Loading feature data...")
    data = pd.read_parquet(FEATURES)
    data["observation_time"] = pd.to_datetime(data["observation_time"])

    print("Loading magnetogram labels...")
    labels = pd.read_csv(LABELS)
    labels["observation_time"] = pd.to_datetime(labels["observation_time"])

    print("\nLoading XGBoost...")
    package = joblib.load(XGB_PATH)
    xgb_model = package["model"]
    imputer = package["imputer"]
    features = package["features"]

    print("Loading CNN...")
    cnn_model, cnn_device = load_cnn()

    # --------------------------------------------------------
    # VALIDATION PERIOD: fit fusion model + pick threshold here.
    # Neither XGBoost nor the CNN were trained on this period,
    # so it's a legitimate place to fit a small meta-learner.
    # --------------------------------------------------------

    val_merged = build_period_probabilities(
        data, labels,
        VALIDATION_START, TEST_START,
        xgb_model, imputer, features,
        cnn_model, cnn_device,
        period_name="VALIDATION",
    )

    y_val = val_merged["target_24h_label"].astype(int)
    xgb_val = val_merged["xgb_probability"].values
    cnn_val = val_merged["cnn_probability"].values

    print("\n" + "=" * 60)
    print(" VALIDATION-PERIOD INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 60)
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_val, xgb_val):.4f}")
    print(f"XGBoost PR-AUC : {average_precision_score(y_val, xgb_val):.4f}")
    print(f"CNN ROC-AUC    : {roc_auc_score(y_val, cnn_val):.4f}")
    print(f"CNN PR-AUC     : {average_precision_score(y_val, cnn_val):.4f}")

    fusion_X_val = np.column_stack([xgb_val, cnn_val])

    print("\nFitting fusion model on VALIDATION period only...")
    fusion_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    fusion_model.fit(fusion_X_val, y_val)

    fusion_val_prob = fusion_model.predict_proba(fusion_X_val)[:, 1]

    # Threshold is also chosen on validation only.
    thresholds = np.arange(0.01, 0.51, 0.01)
    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        preds = (fusion_val_prob >= threshold).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print(f"Fusion threshold selected on validation: {best_threshold:.2f}")

    # --------------------------------------------------------
    # TEST PERIOD: the ONLY place we evaluate. Fusion model and
    # threshold are already frozen from the validation step.
    # --------------------------------------------------------

    test_merged = build_period_probabilities(
        data, labels,
        TEST_START, None,
        xgb_model, imputer, features,
        cnn_model, cnn_device,
        period_name="TEST",
    )

    y_test = test_merged["target_24h_label"].astype(int)
    xgb_test = test_merged["xgb_probability"].values
    cnn_test = test_merged["cnn_probability"].values

    print("\n" + "=" * 60)
    print(" TEST-PERIOD INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 60)
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, xgb_test):.4f}")
    print(f"XGBoost PR-AUC : {average_precision_score(y_test, xgb_test):.4f}")
    print(f"CNN ROC-AUC    : {roc_auc_score(y_test, cnn_test):.4f}")
    print(f"CNN PR-AUC     : {average_precision_score(y_test, cnn_test):.4f}")

    fusion_X_test = np.column_stack([xgb_test, cnn_test])
    fusion_test_prob = fusion_model.predict_proba(fusion_X_test)[:, 1]

    fusion_roc = roc_auc_score(y_test, fusion_test_prob)
    fusion_pr = average_precision_score(y_test, fusion_test_prob)

    final_predictions = (fusion_test_prob >= best_threshold).astype(int)

    precision = precision_score(y_test, final_predictions, zero_division=0)
    recall = recall_score(y_test, final_predictions, zero_division=0)
    final_f1 = f1_score(y_test, final_predictions, zero_division=0)
    cm = confusion_matrix(y_test, final_predictions)

    print("\n" + "=" * 60)
    print(" FUSION RESULTS (evaluated on unseen TEST period only)")
    print("=" * 60)
    print(f"ROC-AUC  : {fusion_roc:.4f}")
    print(f"PR-AUC   : {fusion_pr:.4f}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {final_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    test_merged["fusion_probability"] = fusion_test_prob
    test_merged["fusion_prediction"] = final_predictions
    test_merged.to_csv(OUTPUT_DIR / "fusion_predictions.csv", index=False)

    joblib.dump(
        {
            "model": fusion_model,
            "features": ["xgb_probability", "cnn_probability"],
            "threshold": best_threshold,
        },
        OUTPUT_DIR / "fusion_model.joblib"
    )

    results = pd.DataFrame([
        {
            "model": "XGBoost",
            "roc_auc": roc_auc_score(y_test, xgb_test),
            "pr_auc": average_precision_score(y_test, xgb_test),
        },
        {
            "model": "CNN",
            "roc_auc": roc_auc_score(y_test, cnn_test),
            "pr_auc": average_precision_score(y_test, cnn_test),
        },
        {
            "model": "Fusion",
            "roc_auc": fusion_roc,
            "pr_auc": fusion_pr,
            "precision": precision,
            "recall": recall,
            "f1": final_f1,
            "threshold": best_threshold,
        },
    ])

    results.to_csv(OUTPUT_DIR / "fusion_results.csv", index=False)

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()