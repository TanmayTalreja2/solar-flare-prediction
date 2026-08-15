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
from xgboost import XGBClassifier


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_03.parquet"
)

MODEL_PATH = Path(
    "models/xgboost_30d_baseline.joblib"
)


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


def load_data() -> pd.DataFrame:

    print("Loading feature dataset...")

    data = pd.read_parquet(
        INPUT_PATH
    )

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data = data.sort_values(
        "observation_time"
    ).reset_index(
        drop=True
    )

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    return data


def chronological_split(
    data: pd.DataFrame,
):

    split_index = int(
        len(data) * 0.80
    )

    train = data.iloc[
        :split_index
    ].copy()

    test = data.iloc[
        split_index:
    ].copy()

    return train, test


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
):

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


def create_model(
    y_train: pd.Series,
):

    positive = int(
        y_train.sum()
    )

    negative = int(
        len(y_train) - positive
    )

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.3f}"
    )

    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )


def evaluate(
    model,
    X_test,
    y_test,
):

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
    print(
        "========== MODEL EVALUATION =========="
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
    print(
        "========== CONFUSION MATRIX =========="
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print(
        "========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return probabilities


def main():

    print(
        "========================================"
    )

    print(
        " 30-DAY XGBOOST BASELINE"
    )

    print(
        "========================================"
    )

    data = load_data()

    train, test = chronological_split(
        data
    )

    print()
    print(
        "========== CHRONOLOGICAL SPLIT =========="
    )

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows: {len(test)}"
    )

    print(
        f"Training period: "
        f"{train['observation_time'].min()} "
        f"→ "
        f"{train['observation_time'].max()}"
    )

    print(
        f"Testing period: "
        f"{test['observation_time'].min()} "
        f"→ "
        f"{test['observation_time'].max()}"
    )

    print()
    print(
        "Training target:"
    )

    print(
        train[TARGET]
        .value_counts()
    )

    print()
    print(
        "Testing target:"
    )

    print(
        test[TARGET]
        .value_counts()
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_features(
        train,
        test,
    )

    print()
    print(
        "========== IMPUTATION =========="
    )

    print(
        "Strategy: median"
    )

    print(
        "Imputer fitted only on training data."
    )

    model = create_model(
        y_train
    )

    print()
    print(
        "========== MODEL =========="
    )

    print(
        "Training XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    evaluate(
        model,
        X_test,
        y_test,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print(
        "========== MODEL SAVED =========="
    )

    print(
        f"Output: {MODEL_PATH}"
    )

    print()
    print(
        "========== 30-DAY BASELINE COMPLETE =========="
    )


if __name__ == "__main__":
    main()