from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)

MODEL_PATH = Path(
    "models/"
    "xgboost_2012_temporal_features.joblib"
)


# ============================================================
# ORIGINAL FEATURES
# ============================================================
#
# These are the features already present before temporal
# feature engineering.
#
# 5 SHARP physical features
# 4 logarithmic transformations
# 2 time features
#
# Total = 11
# ============================================================

ORIGINAL_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",

    "LOG_USFLUX",
    "LOG_TOTUSJH",
    "LOG_TOTPOT",
    "LOG_MEANPOT",

    "observation_hour",
    "day_of_year",
]


# ============================================================
# TEMPORAL FEATURES
# ============================================================
#
# 5 base features
# × 4 time windows
# × 3 temporal operations
#
# = 60 temporal features
#
# CHANGE       = absolute change
# RELCHANGE   = relative change
# ROLLSTD     = rolling standard deviation
#
# Total temporal features = 60
# ============================================================

BASE_TEMPORAL_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]


TEMPORAL_WINDOWS = [
    "1h",
    "3h",
    "6h",
    "12h",
]


TEMPORAL_OPERATIONS = [
    "CHANGE",
    "RELCHANGE",
    "ROLLSTD",
]


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "target_24h"


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================
#
# January-June -> training
# July-December -> testing
#
# This is important because this is a temporal prediction
# problem. We do NOT randomly shuffle the observations.
# ============================================================

TRAIN_END = pd.Timestamp(
    "2012-06-30 23:59:59"
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the engineered temporal feature dataset.

    Returns
    -------
    pandas.DataFrame
        Complete dataset containing original features,
        temporal features and target.
    """

    print("========================================")
    print(" FULL 2012 TEMPORAL XGBOOST")
    print("========================================")

    print("Loading temporal feature dataset...")

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


# ============================================================
# BUILD FEATURE LIST
# ============================================================

def get_temporal_features():
    """
    Generate the names of all 60 temporal features.

    5 physical features
    × 4 time windows
    × 3 operations
    = 60 features

    Returns
    -------
    list
        List containing all temporal feature names.
    """

    temporal_features = []

    for feature in BASE_TEMPORAL_FEATURES:

        for operation in TEMPORAL_OPERATIONS:

            for window in TEMPORAL_WINDOWS:

                name = (
                    f"{feature}_"
                    f"{operation}_"
                    f"{window}"
                )

                temporal_features.append(name)

    return temporal_features


# ============================================================
# VALIDATE FEATURES
# ============================================================

def prepare_features(data):
    """
    Validate that every feature required by the model
    exists in the dataset.

    Returns
    -------
    list
        Complete feature column list.
    """

    print()
    print("========== FEATURE VALIDATION ==========")

    temporal_features = get_temporal_features()

    feature_columns = (
        ORIGINAL_FEATURES
        + temporal_features
    )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    print(
        f"Original features: "
        f"{len(ORIGINAL_FEATURES)}"
    )

    print(
        f"Temporal features: "
        f"{len(temporal_features)}"
    )

    print(
        f"Total features used: "
        f"{len(feature_columns)}"
    )

    if missing_features:

        print("Missing features:")

        for feature in missing_features:
            print(f"  {feature}")

        raise ValueError(
            "Required features are missing "
            "from dataset."
        )

    print("All required features found.")

    return feature_columns


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    data,
    feature_columns,
):
    """
    Split the dataset chronologically.

    Training:
        January 2012 -> June 2012

    Testing:
        July 2012 -> December 2012

    No future information is allowed to enter training.
    """

    print()
    print("========== CHRONOLOGICAL SPLIT ==========")

    data = data.copy()

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data = data.sort_values(
        "observation_time"
    ).reset_index(drop=True)

    train_mask = (
        data["observation_time"]
        <= TRAIN_END
    )

    test_mask = ~train_mask

    train_data = data.loc[
        train_mask
    ].copy()

    test_data = data.loc[
        test_mask
    ].copy()

    print(
        f"Training rows: {len(train_data)}"
    )

    print(
        f"Testing rows:  {len(test_data)}"
    )

    print()
    print("Training period:")

    print(
        f"{train_data['observation_time'].min()}"
        f" → "
        f"{train_data['observation_time'].max()}"
    )

    print()
    print("Testing period:")

    print(
        f"{test_data['observation_time'].min()}"
        f" → "
        f"{test_data['observation_time'].max()}"
    )

    X_train = train_data[
        feature_columns
    ].copy()

    X_test = test_data[
        feature_columns
    ].copy()

    y_train = train_data[
        TARGET_COLUMN
    ].astype(int)

    y_test = test_data[
        TARGET_COLUMN
    ].astype(int)

    print()
    print("Training target:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    print()
    print("Testing target:")

    print(
        y_test.value_counts()
        .sort_index()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_data,
        test_data,
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_training_data(
    X_train,
    X_test,
):
    """
    Handle missing values using median imputation.

    IMPORTANT:
    The imputer is fitted ONLY on training data.

    This prevents information from the testing period
    leaking into the training process.
    """

    print()
    print("========== PREPARING FEATURES ==========")

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"X_test shape:  "
        f"{X_test.shape}"
    )

    print()
    print("========== IMPUTATION ==========")

    print(
        "Strategy: median"
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test_imputed = (
        imputer.transform(
            X_test
        )
    )

    print(
        "Imputer fitted only on "
        "training data."
    )

    return (
        X_train_imputed,
        X_test_imputed,
        imputer,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):
    """
    Train the XGBoost classifier.

    Because flare observations are rare, scale_pos_weight
    is calculated from the training data.
    """

    print()
    print("========== MODEL ==========")

    negative_samples = (
        y_train == 0
    ).sum()

    positive_samples = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_samples
        / positive_samples
    )

    print(
        f"Negative samples: "
        f"{negative_samples}"
    )

    print(
        f"Positive samples: "
        f"{positive_samples}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.3f}"
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    print()
    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    feature_columns,
):
    """
    Evaluate the trained model on the completely unseen
    July-December 2012 test period.

    Metrics:
        ROC-AUC
        PR-AUC
        Precision
        Recall
        F1
        Confusion Matrix
        Classification Report
    """

    print()
    print("========== TEMPORAL EVALUATION ==========")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print()
    print("========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(cm)

    print()
    print("========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    feature_columns,
):
    """
    Display the most important features according to
    XGBoost.

    This helps explain which SHARP and temporal features
    contribute most to the prediction.
    """

    print()
    print("========== TOP TEMPORAL FEATURES ==========")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    )

    importance = (
        importance
        .sort_values(
            ascending=False
        )
    )

    print(
        importance.head(20)
    )

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    imputer,
    feature_columns,
    metrics,
):
    """
    Save everything required for future prediction.

    We save:
        model
        imputer
        exact feature order
        evaluation metrics
        prediction threshold

    Saving feature_columns is extremely important because
    the prediction script must use exactly the same feature
    order as training.
    """

    print()
    print("========== MODEL SAVING ==========")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": model,

        "imputer": imputer,

        "features": feature_columns,

        "target": TARGET_COLUMN,

        "threshold": 0.5,

        "metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },

        "training_period": (
            "2012-01-01 to 2012-06-30"
        ),

        "testing_period": (
            "2012-07-01 to 2012-12-31"
        ),

        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    print(
        "Model saved:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Execute the complete training pipeline.

    Pipeline:

        Load data
            ↓
        Build feature list
            ↓
        Validate features
            ↓
        Chronological split
            ↓
        Median imputation
            ↓
        XGBoost training
            ↓
        Temporal evaluation
            ↓
        Feature importance
            ↓
        Save model
    """

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # 2. Build and validate feature list
    # --------------------------------------------------------

    feature_columns = prepare_features(
        data
    )

    # --------------------------------------------------------
    # 3. Chronological train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_data,
        test_data,
    ) = chronological_split(
        data,
        feature_columns,
    )

    # --------------------------------------------------------
    # 4. Impute missing values
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        imputer,
    ) = prepare_training_data(
        X_train,
        X_test,
    )

    # --------------------------------------------------------
    # 5. Train XGBoost
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        feature_columns,
    )

    # --------------------------------------------------------
    # 7. Feature importance
    # --------------------------------------------------------

    importance = show_feature_importance(
        model,
        feature_columns,
    )

    # --------------------------------------------------------
    # 8. Save everything
    # --------------------------------------------------------

    save_model(
        model,
        imputer,
        feature_columns,
        metrics,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )

    print(
        " FULL 2012 TEMPORAL FEATURE MODEL COMPLETE"
    )

    print(
        "========================================"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()