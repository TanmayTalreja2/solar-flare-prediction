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

# Same chronological test period
TEST_START = pd.Timestamp(
    "2012-07-01 00:00:00"
)


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


def get_cnn_predictions(
    labels,
):

    import torch

    model, device = load_cnn()

    probabilities = []

    print(
        "Generating CNN predictions..."
    )

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

        tensor = tensor.to(device)

        with torch.no_grad():

            output = model(
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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" XGBOOST + CNN FUSION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "Loading feature data..."
    )

    data = pd.read_parquet(
        FEATURES
    )

    data[
        "observation_time"
    ] = pd.to_datetime(
        data["observation_time"]
    )

    test_data = data[
        data["observation_time"]
        >= TEST_START
    ].copy()

    print(
        f"Test observations: "
        f"{len(test_data)}"
    )

    print(
        "Loading magnetogram labels..."
    )

    labels = pd.read_csv(
        LABELS
    )

    labels[
        "observation_time"
    ] = pd.to_datetime(
        labels["observation_time"]
    )

    labels = labels[
        labels["observation_time"]
        >= TEST_START
    ].copy()

    print(
        f"Test magnetograms: "
        f"{len(labels)}"
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print(
        "\nLoading XGBoost..."
    )

    package = joblib.load(
        XGB_PATH
    )

    xgb_model = package["model"]
    imputer = package["imputer"]
    features = package["features"]

    # Match by HARPNUM + observation time
    xgb_subset = test_data[
        [
            "HARPNUM",
            "NOAA_AR",
            "observation_time",
            "target_24h",
        ]
        + features
    ].copy()

    merged = labels.merge(
        xgb_subset,
        on=[
            "HARPNUM",
            "observation_time",
        ],
        how="inner",
        suffixes=("_label", "")
    )

    print(
        f"Matched observations: "
        f"{len(merged)}"
    )

    # --------------------------------------------------------
    # XGBoost probabilities
    # --------------------------------------------------------

    X = merged[
        features
    ]

    X_imp = imputer.transform(
        X
    )

    merged[
        "xgb_probability"
    ] = xgb_model.predict_proba(
        X_imp
    )[:, 1]

    # --------------------------------------------------------
    # CNN probabilities
    # --------------------------------------------------------

    cnn_probs = get_cnn_predictions(
        merged[
            [
                "HARPNUM",
                "observation_time",
                "target_24h_label",
            ]
        ].rename(
            columns={
                "target_24h_label":
                    "target_24h"
            }
        )
    )

    merged[
        "cnn_probability"
    ] = cnn_probs

    # Remove missing CNN predictions
    merged = merged.dropna(
        subset=[
            "xgb_probability",
            "cnn_probability",
        ]
    ).reset_index(
        drop=True
    )

    y = merged[
        "target_24h_label"
    ].astype(int)

    print(
        f"\nFinal matched samples: "
        f"{len(merged)}"
    )

    print(
        f"Positive: {(y == 1).sum()}"
    )

    print(
        f"Negative: {(y == 0).sum()}"
    )

    # --------------------------------------------------------
    # Individual models
    # --------------------------------------------------------

    xgb_prob = merged[
        "xgb_probability"
    ].values

    cnn_prob = merged[
        "cnn_probability"
    ].values

    print("\n" + "=" * 60)
    print(" INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"XGBoost ROC-AUC: "
        f"{roc_auc_score(y, xgb_prob):.4f}"
    )

    print(
        f"XGBoost PR-AUC : "
        f"{average_precision_score(y, xgb_prob):.4f}"
    )

    print(
        f"CNN ROC-AUC    : "
        f"{roc_auc_score(y, cnn_prob):.4f}"
    )

    print(
        f"CNN PR-AUC     : "
        f"{average_precision_score(y, cnn_prob):.4f}"
    )

    # --------------------------------------------------------
    # Fusion model
    # --------------------------------------------------------

    fusion_X = np.column_stack(
        [
            xgb_prob,
            cnn_prob,
        ]
    )

    print("\nTraining fusion model...")

    fusion_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    fusion_model.fit(
        fusion_X,
        y
    )

    fusion_prob = fusion_model.predict_proba(
        fusion_X
    )[:, 1]

    # --------------------------------------------------------
    # Fusion evaluation
    # --------------------------------------------------------

    fusion_roc = roc_auc_score(
        y,
        fusion_prob
    )

    fusion_pr = average_precision_score(
        y,
        fusion_prob
    )

    # Threshold search
    thresholds = np.arange(
        0.01,
        0.51,
        0.01
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            fusion_prob
            >= threshold
        ).astype(int)

        score = f1_score(
            y,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    final_predictions = (
        fusion_prob
        >= best_threshold
    ).astype(int)

    precision = precision_score(
        y,
        final_predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        final_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        final_predictions
    )

    print("\n" + "=" * 60)
    print(" FUSION RESULTS")
    print("=" * 60)

    print(
        f"ROC-AUC : {fusion_roc:.4f}"
    )

    print(
        f"PR-AUC  : {fusion_pr:.4f}"
    )

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {best_f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    merged[
        "fusion_probability"
    ] = fusion_prob

    merged[
        "fusion_prediction"
    ] = final_predictions

    merged.to_csv(
        OUTPUT_DIR
        / "fusion_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": fusion_model,
            "features": [
                "xgb_probability",
                "cnn_probability",
            ],
            "threshold":
                best_threshold,
        },
        OUTPUT_DIR
        / "fusion_model.joblib"
    )

    results = pd.DataFrame([

        {
            "model": "XGBoost",
            "roc_auc":
                roc_auc_score(
                    y,
                    xgb_prob
                ),
            "pr_auc":
                average_precision_score(
                    y,
                    xgb_prob
                ),
        },

        {
            "model": "CNN",
            "roc_auc":
                roc_auc_score(
                    y,
                    cnn_prob
                ),
            "pr_auc":
                average_precision_score(
                    y,
                    cnn_prob
                ),
        },

        {
            "model": "Fusion",
            "roc_auc":
                fusion_roc,
            "pr_auc":
                fusion_pr,
            "precision":
                precision,
            "recall":
                recall,
            "f1":
                best_f1,
            "threshold":
                best_threshold,
        },

    ])

    results.to_csv(
        OUTPUT_DIR
        / "fusion_results.csv",
        index=False
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()