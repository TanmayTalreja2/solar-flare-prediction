from pathlib import Path

import joblib
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
    "sharp_goes_temporal_features_2012_full.parquet"
)

MODEL_PATH = Path(
    "models/xgboost_2012_temporal.joblib"
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


def load_data():

    print("Loading full-year feature dataset...")

    data = pd.read_parquet(
        INPUT_PATH
    ).copy()

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data = data.sort_values(
        "observation_time"
    ).reset_index(drop=True)

    print(
        f"Rows: {len(data)}"
    )

    return data


def create_temporal_split(data):

    cutoff = pd.Timestamp(
        "2012-07-01 00:00:00"
    )

    train = data[
        data["observation_time"] < cutoff
    ].copy()

    test = data[
        data["observation_time"] >= cutoff
    ].copy()

    return train, test


def prepare_features(train, test):

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
        imputer,
    )


def create_model(y_train):

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
        n_estimators=300,
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


def evaluate(model, X_test, y_test):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print()
    print(
        "========== TEMPORAL EVALUATION =========="
    )

    print(
        f"ROC-AUC : "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"PR-AUC  : "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1-score : "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print()
    print(
        "========== CONFUSION MATRIX =========="
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
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
            labels=[0, 1],
            zero_division=0,
        )
    )

    return probabilities


def main():

    print(
        "========================================"
    )
    print(
        " FULL 2012 TEMPORAL XGBOOST"
    )
    print(
        "========================================"
    )

    data = load_data()

    train, test = create_temporal_split(
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
        f"Testing rows:  {len(test)}"
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
    print("Training target:")
    print(
        train[TARGET].value_counts()
    )

    print()
    print("Testing target:")
    print(
        test[TARGET].value_counts()
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        imputer,
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

    probabilities = evaluate(
        model,
        X_test,
        y_test,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "features": FEATURES,
        },
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
        "========== FULL 2012 TEMPORAL MODEL COMPLETE =========="
    )


if __name__ == "__main__":
    main()