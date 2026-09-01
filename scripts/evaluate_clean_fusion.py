"""
Clean XGBoost + CNN Fusion Evaluation

Purpose:
    Evaluate XGBoost + CNN fusion without leaking final-test information.

Temporal split:
    TRAIN      : Jan-Apr 2012
    VALIDATION : May-Jun 2012
    FINAL TEST : Jul-Dec 2012

Procedure:
    1. Load temporal XGBoost model.
    2. Load trained CNN.
    3. Generate CNN predictions for all magnetograms.
    4. Match CNN observations with XGBoost observations.
    5. Use VALIDATION only to:
          - select fusion weight
          - select threshold for recall >= 80%
    6. Freeze weight + threshold.
    7. Evaluate once on FINAL TEST.
    8. Save final predictions and metrics.
"""

from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
# ============================================================
# PROJECT PATH + DEVICE
# ============================================================

project_root = Path(__file__).parent.parent

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "sharp_goes_temporal_features_2012_full.parquet"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
    / "dataset_labels.csv"
)

MAGNETOGRAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
)

XGB_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_2012_temporal_features.joblib"
)

CNN_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn_magnetogram.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ensemble"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32

TARGET_RECALL = 0.80

# Candidate CNN weights.
# Fusion probability:
#
#     fusion = xgb_weight * xgb_probability
#            + cnn_weight * cnn_probability
#
WEIGHTS = np.arange(
    0.0,
    1.01,
    0.05
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print(" CLEAN XGBOOST + CNN FUSION")
print("=" * 60)

print()
print(
    f"Using device: {DEVICE}"
)


# ============================================================
# CNN DATASET
# ============================================================

class MagnetogramDataset(Dataset):

    def __init__(
        self,
        dataframe,
        magnetogram_dir,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.magnetogram_dir = (
            Path(magnetogram_dir)
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        file_name = row["file_name"]

        file_path = (
            self.magnetogram_dir
            / file_name
        )

        data = np.load(
            file_path
        )

        image = data["img"].astype(
            np.float32
        )

        image = torch.from_numpy(
            image
        )

        # [H, W] -> [1, H, W]

        if image.ndim == 2:

            image = image.unsqueeze(0)

        return (
            image,
            index
        )


# ============================================================
# CNN MODEL
#
# This architecture matches the improved CNN training script.
# ============================================================

class SolarFlareCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.10),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.15),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.20),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        x
    ):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD CNN
# ============================================================

def build_cnn_model():
    model = models.resnet18(weights=None)

    # Magnetogram is single-channel
    model.conv1 = nn.Conv2d(
        1,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Same classifier used during training
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 1)
    )

    return model


def load_cnn():
    print("Loading CNN...")

    model = build_cnn_model()

    checkpoint = torch.load(
        project_root / "models" / "cnn_magnetogram.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("CNN loaded successfully.")

    return model


# ============================================================
# GENERATE CNN PREDICTIONS
# ============================================================

def generate_cnn_predictions(
    model,
    dataframe,
):

    dataset = MagnetogramDataset(
        dataframe,
        MAGNETOGRAM_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    probabilities = np.zeros(
        len(dataframe),
        dtype=np.float32
    )

    print()
    print(
        "Generating CNN predictions..."
    )

    with torch.no_grad():

        for images, indices in loader:

            images = images.to(
                DEVICE
            )

            logits = model(
                images
            ).squeeze(1)

            probs = torch.sigmoid(
                logits
            )

            probabilities[
                indices.numpy()
            ] = (
                probs.cpu()
                .numpy()
            )

    return probabilities


# ============================================================
# LOAD DATA
# ============================================================

print()
print(
    "Loading feature data..."
)

features = pd.read_parquet(
    FEATURE_PATH
)

features[
    "observation_time"
] = pd.to_datetime(
    features[
        "observation_time"
    ]
)

print(
    f"Total observations: "
    f"{len(features)}"
)


print()
print(
    "Loading magnetogram labels..."
)

labels = pd.read_csv(
    LABEL_PATH
)

labels[
    "observation_time"
] = pd.to_datetime(
    labels[
        "observation_time"
    ]
)

print(
    f"Total magnetogram labels: "
    f"{len(labels)}"
)


# ============================================================
# RECONSTRUCT MAGNETOGRAM FILENAMES
# ============================================================

def build_filename(row):

    harp = int(
        row["HARPNUM"]
    )

    timestamp = (
        row["observation_time"]
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    target = int(
        row["target_24h"]
    )

    return (
        f"harp_{harp}_"
        f"{timestamp}_"
        f"t{target}.npz"
    )


labels["file_name"] = labels.apply(
    build_filename,
    axis=1
)


# ============================================================
# CHECK AVAILABLE FILES
# ============================================================

labels["file_exists"] = (
    labels["file_name"]
    .apply(
        lambda x:
        (
            MAGNETOGRAM_DIR
            / x
        ).exists()
    )
)

labels = labels[
    labels["file_exists"]
].copy()

print()
print(
    f"Valid magnetograms: "
    f"{len(labels)}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_end = pd.Timestamp(
    "2012-04-30 23:59:59"
)

validation_end = pd.Timestamp(
    "2012-06-30 23:59:59"
)

train_labels = labels[
    labels["observation_time"]
    <= train_end
].copy()

validation_labels = labels[
    (
        labels["observation_time"]
        > train_end
    )
    &
    (
        labels["observation_time"]
        <= validation_end
    )
].copy()

test_labels = labels[
    labels["observation_time"]
    > validation_end
].copy()


print()
print("=" * 60)
print(" CHRONOLOGICAL SPLIT")
print("=" * 60)

print()
print(
    f"TRAIN      : {len(train_labels)}"
)

print(
    f"VALIDATION : {len(validation_labels)}"
)

print(
    f"FINAL TEST : {len(test_labels)}"
)


# ============================================================
# LOAD MODELS
# ============================================================

print()
print(
    "Loading XGBoost..."
)

xgb_package = joblib.load(
    XGB_PATH
)

xgb_model = xgb_package[
    "model"
]

xgb_imputer = xgb_package[
    "imputer"
]

xgb_features = xgb_package[
    "features"
]

print(
    "XGBoost loaded successfully."
)


cnn_model = load_cnn()


# ============================================================
# GENERATE CNN PREDICTIONS FOR ALL MAGNETOGRAMS
# ============================================================

cnn_probabilities = (
    generate_cnn_predictions(
        cnn_model,
        labels
    )
)

labels[
    "cnn_probability"
] = cnn_probabilities


# ============================================================
# MATCH WITH XGBOOST FEATURES
# ============================================================

print()
print(
    "Matching CNN observations "
    "with XGBoost data..."
)

# Keep the matching keys PLUS all features
# required by the trained XGBoost model.

required_columns = [
    "NOAA_AR",
    "observation_time",
] + [
    feature
    for feature in xgb_features
    if feature not in [
        "NOAA_AR",
        "observation_time",
    ]
]

features_for_merge = features[
    required_columns
].copy()

# Remove duplicate observations before merging.

features_for_merge = (
    features_for_merge
    .drop_duplicates(
        subset=[
            "NOAA_AR",
            "observation_time",
        ],
        keep="first",
    )
)

merged = labels.merge(
    features_for_merge,
    on=[
        "NOAA_AR",
        "observation_time",
    ],
    how="inner",
)

print(
    f"Matched observations: "
    f"{len(merged)}"
)

merged = (
    merged
    .sort_values(
        "observation_time"
    )
    .drop_duplicates(
        subset=[
            "NOAA_AR",
            "observation_time",
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)

print(
    f"After duplicate removal: "
    f"{len(merged)}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

merged = (
    merged
    .sort_values(
        "observation_time"
    )
    .drop_duplicates(
        subset=[
            "NOAA_AR",
            "observation_time",
        ],
        keep="first"
    )
    .reset_index(
        drop=True
    )
)


print(
    f"After duplicate removal: "
    f"{len(merged)}"
)


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

print()
print(
    "Generating XGBoost predictions..."
)


missing_features = [
    feature
    for feature in xgb_features
    if feature not in merged.columns
]

if missing_features:

    raise ValueError(
        "Missing XGBoost features: "
        + str(
            missing_features
        )
    )


X = merged[
    xgb_features
].copy()


X_imputed = (
    xgb_imputer.transform(X)
)


merged[
    "xgb_probability"
] = (
    xgb_model
    .predict_proba(
        X_imputed
    )[:, 1]
)


# ============================================================
# SPLIT MERGED DATA
# ============================================================

train = merged[
    merged["observation_time"]
    <= train_end
].copy()

validation = merged[
    (
        merged["observation_time"]
        > train_end
    )
    &
    (
        merged["observation_time"]
        <= validation_end
    )
].copy()

test = merged[
    merged["observation_time"]
    > validation_end
].copy()


print()
print("=" * 60)
print(" MATCHED TEMPORAL DATA")
print("=" * 60)

print()
print(
    f"Train      : {len(train)}"
)

print(
    f"Validation : {len(validation)}"
)

print(
    f"Final Test : {len(test)}"
)


# ============================================================
# FUSION FUNCTION
# ============================================================

def fuse_predictions(
    dataframe,
    xgb_weight
):

    cnn_weight = (
        1.0
        - xgb_weight
    )

    return (
        xgb_weight
        * dataframe[
            "xgb_probability"
        ].to_numpy()
        +
        cnn_weight
        * dataframe[
            "cnn_probability"
        ].to_numpy()
    )


# ============================================================
# VALIDATION WEIGHT SEARCH
# ============================================================

print()
print("=" * 60)
print(" VALIDATION FUSION WEIGHT SEARCH")
print("=" * 60)


validation_y = (
    validation[
        "target_24h"
    ]
    .astype(int)
    .to_numpy()
)


weight_results = []


for xgb_weight in WEIGHTS:

    cnn_weight = (
        1.0
        - xgb_weight
    )

    probabilities = (
        fuse_predictions(
            validation,
            xgb_weight
        )
    )

    roc_auc = roc_auc_score(
        validation_y,
        probabilities
    )

    pr_auc = (
        average_precision_score(
            validation_y,
            probabilities
        )
    )

    weight_results.append({

        "xgb_weight":
            xgb_weight,

        "cnn_weight":
            cnn_weight,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

    })

    print(
        f"XGB={xgb_weight:.2f} | "
        f"CNN={cnn_weight:.2f} | "
        f"ROC-AUC={roc_auc:.4f} | "
        f"PR-AUC={pr_auc:.4f}"
    )


weight_df = pd.DataFrame(
    weight_results
)


# ============================================================
# SELECT WEIGHT
#
# Primary objective:
#     highest validation PR-AUC
#
# Tie-break:
#     highest ROC-AUC
# ============================================================

best_weight_row = (
    weight_df
    .sort_values(
        [
            "pr_auc",
            "roc_auc",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .iloc[0]
)


BEST_XGB_WEIGHT = float(
    best_weight_row[
        "xgb_weight"
    ]
)

BEST_CNN_WEIGHT = float(
    best_weight_row[
        "cnn_weight"
    ]
)


print()
print("=" * 60)
print(" BEST VALIDATION FUSION")
print("=" * 60)

print(
    f"XGBoost weight : "
    f"{BEST_XGB_WEIGHT:.2f}"
)

print(
    f"CNN weight     : "
    f"{BEST_CNN_WEIGHT:.2f}"
)

print(
    f"Validation ROC-AUC : "
    f"{best_weight_row['roc_auc']:.4f}"
)

print(
    f"Validation PR-AUC  : "
    f"{best_weight_row['pr_auc']:.4f}"
)


# ============================================================
# VALIDATION THRESHOLD SEARCH
#
# Requirement:
#     Recall >= 80%
#
# Among eligible thresholds:
#     maximize precision
#     then maximize F1
# ============================================================

validation_probabilities = (
    fuse_predictions(
        validation,
        BEST_XGB_WEIGHT
    )
)


threshold_results = []


for threshold in np.arange(
    0.01,
    0.96,
    0.01
):

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


    precision = precision_score(
        validation_y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        validation_y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        validation_y,
        predictions,
        zero_division=0
    )


    threshold_results.append({

        "threshold":
            round(
                threshold,
                2
            ),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

    })


threshold_df = pd.DataFrame(
    threshold_results
)


eligible_thresholds = (
    threshold_df[
        threshold_df["recall"]
        >= TARGET_RECALL
    ]
    .copy()
)


if len(
    eligible_thresholds
) == 0:

    print()
    print(
        "WARNING:"
    )

    print(
        "No validation threshold "
        "achieved 80% recall."
    )

    print(
        "Selecting threshold with "
        "highest recall."
    )

    best_threshold_row = (
        threshold_df
        .sort_values(
            [
                "recall",
                "precision",
                "f1",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )
        .iloc[0]
    )

else:

    best_threshold_row = (
        eligible_thresholds
        .sort_values(
            [
                "precision",
                "f1",
                "recall",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )
        .iloc[0]
    )


BEST_THRESHOLD = float(
    best_threshold_row[
        "threshold"
    ]
)


validation_predictions = (
    validation_probabilities
    >= BEST_THRESHOLD
).astype(int)


validation_precision = (
    precision_score(
        validation_y,
        validation_predictions,
        zero_division=0
    )
)

validation_recall = (
    recall_score(
        validation_y,
        validation_predictions,
        zero_division=0
    )
)

validation_f1 = (
    f1_score(
        validation_y,
        validation_predictions,
        zero_division=0
    )
)


print()
print("=" * 60)
print(" VALIDATION THRESHOLD")
print("=" * 60)

print(
    f"Target recall : "
    f"{TARGET_RECALL:.2f}"
)

print(
    f"Threshold     : "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Precision     : "
    f"{validation_precision:.4f}"
)

print(
    f"Recall        : "
    f"{validation_recall:.4f}"
)

print(
    f"F1            : "
    f"{validation_f1:.4f}"
)


# ============================================================
# FINAL UNSEEN TEST
#
# WEIGHT AND THRESHOLD ARE NOW FROZEN.
# ============================================================

print()
print("=" * 60)
print(" FINAL UNSEEN TEST EVALUATION")
print("=" * 60)

test_y = (
    test[
        "target_24h"
    ]
    .astype(int)
    .to_numpy()
)


test_probabilities = (
    fuse_predictions(
        test,
        BEST_XGB_WEIGHT
    )
)


test_predictions = (
    test_probabilities
    >= BEST_THRESHOLD
).astype(int)


test_roc_auc = (
    roc_auc_score(
        test_y,
        test_probabilities
    )
)

test_pr_auc = (
    average_precision_score(
        test_y,
        test_probabilities
    )
)

test_precision = (
    precision_score(
        test_y,
        test_predictions,
        zero_division=0
    )
)

test_recall = (
    recall_score(
        test_y,
        test_predictions,
        zero_division=0
    )
)

test_f1 = (
    f1_score(
        test_y,
        test_predictions,
        zero_division=0
    )
)

test_cm = confusion_matrix(
    test_y,
    test_predictions
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print(
    "FINAL TEST PERIOD:"
)

print(
    "July 1 - December 31, 2012"
)

print()
print(
    f"ROC-AUC  : "
    f"{test_roc_auc:.4f}"
)

print(
    f"PR-AUC   : "
    f"{test_pr_auc:.4f}"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1        : "
    f"{test_f1:.4f}"
)

print()
print(
    "CONFUSION MATRIX"
)

print(
    test_cm
)


# ============================================================
# SAVE FINAL PREDICTIONS
# ============================================================

prediction_output = test[
    [
        "HARPNUM",
        "NOAA_AR",
        "observation_time",
        "T_REC",
        "target_24h",
    ]
].copy()


prediction_output[
    "xgb_probability"
] = test[
    "xgb_probability"
].to_numpy()


prediction_output[
    "cnn_probability"
] = test[
    "cnn_probability"
].to_numpy()


prediction_output[
    "fusion_probability"
] = test_probabilities


prediction_output[
    "prediction"
] = test_predictions


prediction_path = (
    OUTPUT_DIR
    / "clean_final_fusion_predictions.csv"
)


prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "metric": [

        "xgb_weight",

        "cnn_weight",

        "threshold",

        "validation_precision",

        "validation_recall",

        "validation_f1",

        "final_test_roc_auc",

        "final_test_pr_auc",

        "final_test_precision",

        "final_test_recall",

        "final_test_f1",

    ],

    "value": [

        BEST_XGB_WEIGHT,

        BEST_CNN_WEIGHT,

        BEST_THRESHOLD,

        validation_precision,

        validation_recall,

        validation_f1,

        test_roc_auc,

        test_pr_auc,

        test_precision,

        test_recall,

        test_f1,

    ]

})


metrics_path = (
    OUTPUT_DIR
    / "clean_fusion_metrics.csv"
)


metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE WEIGHT SEARCH
# ============================================================

weight_path = (
    OUTPUT_DIR
    / "validation_fusion_weights.csv"
)

weight_df.to_csv(
    weight_path,
    index=False
)


# ============================================================
# SAVE THRESHOLD SEARCH
# ============================================================

threshold_path = (
    OUTPUT_DIR
    / "validation_fusion_thresholds.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_path = (
    OUTPUT_DIR
    / "final_fusion_configuration.txt"
)


with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL XGBOOST + CNN FUSION CONFIGURATION\n"
    )

    f.write(
        "=========================================\n\n"
    )

    f.write(
        "Temporal split:\n"
    )

    f.write(
        "Train: January-April 2012\n"
    )

    f.write(
        "Validation: May-June 2012\n"
    )

    f.write(
        "Final test: July-December 2012\n\n"
    )

    f.write(
        "Fusion weights selected using validation only.\n"
    )

    f.write(
        f"XGBoost weight: "
        f"{BEST_XGB_WEIGHT:.2f}\n"
    )

    f.write(
        f"CNN weight: "
        f"{BEST_CNN_WEIGHT:.2f}\n"
    )

    f.write(
        f"Threshold: "
        f"{BEST_THRESHOLD:.2f}\n\n"
    )

    f.write(
        "Validation performance:\n"
    )

    f.write(
        f"Precision: "
        f"{validation_precision:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{validation_recall:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{validation_f1:.4f}\n\n"
    )

    f.write(
        "FINAL UNSEEN TEST PERFORMANCE:\n"
    )

    f.write(
        f"ROC-AUC: "
        f"{test_roc_auc:.4f}\n"
    )

    f.write(
        f"PR-AUC: "
        f"{test_pr_auc:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{test_precision:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{test_recall:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{test_f1:.4f}\n\n"
    )

    f.write(
        "Confusion matrix:\n"
    )

    f.write(
        str(test_cm)
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print(" CLEAN FUSION EVALUATION COMPLETE")
print("=" * 60)

print()
print(
    "Final predictions:"
)

print(
    prediction_path
)

print()
print(
    "Final metrics:"
)

print(
    metrics_path
)

print()
print(
    "Final configuration:"
)

print(
    config_path
)