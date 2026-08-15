from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_03_07.parquet"
)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgboost_baseline.joblib"


FEATURES = [
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

TARGET = "target_24h"


def load_dataset() -> pd.DataFrame:
    """Load the engineered feature dataset."""

    print("Loading feature dataset...")

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def prepare_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if TARGET not in data.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create a stratified train/test split."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print("========== DATA SPLIT ==========")

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    print()
    print("Training target:")
    print(y_train.value_counts())

    print()
    print("Testing target:")
    print(y_test.value_counts())

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def impute_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    """Impute missing values using training-set medians."""

    print()
    print("========== IMPUTATION ==========")

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_test_imputed = imputer.transform(
        X_test
    )

    print("Strategy: median")
    print("Imputer fitted only on training data.")

    return (
        X_train_imputed,
        X_test_imputed,
        imputer,
    )


def build_model(
    y_train: pd.Series,
) -> XGBClassifier:
    """Create the baseline XGBoost classifier."""

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    scale_pos_weight = negative / positive

    print()
    print("========== MODEL ==========")
    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.3f}"
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> None:
    """Evaluate the trained model."""

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

    print()
    print("========== MODEL EVALUATION ==========")

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

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


def save_model(
    model: XGBClassifier,
    imputer: SimpleImputer,
) -> None:
    """Save model and preprocessing together."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "imputer": imputer,
        "features": FEATURES,
        "target": TARGET,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print()
    print("========== MODEL SAVED ==========")
    print(f"Output: {MODEL_PATH}")


def main() -> None:
    """Run the complete baseline training pipeline."""

    data = load_dataset()

    X, y = prepare_data(data)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    (
        X_train_imputed,
        X_test_imputed,
        imputer,
    ) = impute_features(
        X_train,
        X_test,
    )

    model = build_model(y_train)

    print()
    print("Training XGBoost...")

    model.fit(
        X_train_imputed,
        y_train,
    )

    print("Training complete.")

    evaluate_model(
        model,
        X_test_imputed,
        y_test,
    )

    save_model(
        model,
        imputer,
    )

    print()
    print(
        "========== BASELINE TRAINING COMPLETE =========="
    )


if __name__ == "__main__":
    main()