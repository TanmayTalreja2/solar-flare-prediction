from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_03_07.parquet"
)

REPORT_DIR = Path("reports/figures")


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
GROUP_COLUMN = "NOAA_AR"


def load_data() -> pd.DataFrame:
    """Load the engineered dataset."""

    print("Loading engineered dataset...")

    data = pd.read_parquet(INPUT_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data = data.sort_values(
        "observation_time"
    ).reset_index(drop=True)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def create_model(
    y_train: pd.Series,
) -> XGBClassifier:
    """Create an XGBoost model with class balancing."""

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)

    if positive == 0:
        raise ValueError(
            "Training set contains no positive samples."
        )

    scale_pos_weight = negative / positive

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


def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit imputation on training data only."""

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    return X_train, X_test


def calculate_metrics(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics."""

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }


def print_metrics(
    metrics: dict[str, float],
) -> None:
    """Print evaluation metrics."""

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC   : {metrics['pr_auc']:.4f}"
    )
    print(
        f"Precision: {metrics['precision']:.4f}"
    )
    print(
        f"Recall   : {metrics['recall']:.4f}"
    )
    print(
        f"F1-score : {metrics['f1']:.4f}"
    )


def save_evaluation_figures(
    y_test: pd.Series,
    probabilities: np.ndarray,
    prefix: str,
) -> None:
    """Save ROC, PR and confusion matrix plots."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ROC curve
    fig, ax = plt.subplots()

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        ax=ax,
    )

    ax.set_title(
        f"ROC Curve - {prefix}"
    )

    fig.savefig(
        REPORT_DIR / f"roc_{prefix}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Precision-recall curve
    fig, ax = plt.subplots()

    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        ax=ax,
    )

    ax.set_title(
        f"Precision-Recall Curve - {prefix}"
    )

    fig.savefig(
        REPORT_DIR / f"pr_{prefix}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix - {prefix}"
    )

    fig.savefig(
        REPORT_DIR / f"confusion_{prefix}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def chronological_evaluation(
    data: pd.DataFrame,
) -> None:
    """Evaluate using earlier observations to predict later observations."""

    print()
    print(
        "========================================"
    )
    print(
        " CHRONOLOGICAL EVALUATION"
    )
    print(
        "========================================"
    )

    split_index = int(
        len(data) * 0.80
    )

    train = data.iloc[
        :split_index
    ].copy()

    test = data.iloc[
        split_index:
    ].copy()

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

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    X_train, X_test = prepare_features(
        X_train,
        X_test,
    )

    model = create_model(
        y_train
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        probabilities,
    )

    print()
    print("Chronological metrics:")

    print_metrics(metrics)

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            probabilities >= 0.5,
            zero_division=0,
        )
    )

    save_evaluation_figures(
        y_test,
        probabilities,
        "chronological",
    )

    train_regions = set(
        train[GROUP_COLUMN]
        .dropna()
        .unique()
    )

    test_regions = set(
        test[GROUP_COLUMN]
        .dropna()
        .unique()
    )

    overlap = (
        train_regions & test_regions
    )

    print()
    print(
        "Active-region overlap:"
    )

    print(
        f"Training regions: "
        f"{len(train_regions)}"
    )

    print(
        f"Testing regions: "
        f"{len(test_regions)}"
    )

    print(
        f"Overlapping regions: "
        f"{len(overlap)}"
    )

    if overlap:
        print(
            "WARNING: Some active regions "
            "appear in both sets."
        )


def group_evaluation(
    data: pd.DataFrame,
) -> None:
    """Evaluate on active regions unseen during training."""

    print()
    print(
        "========================================"
    )
    print(
        " ACTIVE-REGION GROUP EVALUATION"
    )
    print(
        "========================================"
    )

    valid_data = data[
        data[GROUP_COLUMN].notna()
    ].copy()

    groups = valid_data[
        GROUP_COLUMN
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            valid_data,
            valid_data[TARGET],
            groups,
        )
    )

    train = valid_data.iloc[
        train_indices
    ]

    test = valid_data.iloc[
        test_indices
    ]

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows: {len(test)}"
    )

    train_regions = set(
        train[GROUP_COLUMN]
    )

    test_regions = set(
        test[GROUP_COLUMN]
    )

    overlap = (
        train_regions & test_regions
    )

    print(
        f"Training active regions: "
        f"{len(train_regions)}"
    )

    print(
        f"Testing active regions: "
        f"{len(test_regions)}"
    )

    print(
        f"Overlap: {len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Active-region leakage detected."
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    X_train, X_test = prepare_features(
        X_train,
        X_test,
    )

    model = create_model(
        y_train
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        probabilities,
    )

    print()
    print(
        "Unseen active-region metrics:"
    )

    print_metrics(metrics)

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            probabilities >= 0.5,
            zero_division=0,
        )
    )

    save_evaluation_figures(
        y_test,
        probabilities,
        "active_region_group",
    )


def main() -> None:
    """Run all evaluation strategies."""

    data = load_data()

    chronological_evaluation(
        data
    )

    group_evaluation(
        data
    )

    print()
    print(
        "========================================"
    )
    print(
        " EVALUATION COMPLETE"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()