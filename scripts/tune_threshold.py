from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


MODEL_PATH = Path(
    "models/xgboost_2012_temporal_features.joblib"
)

PREDICTIONS_PATH = Path(
    "results/temporal_model/test_predictions.csv"
)

OUTPUT_PATH = Path(
    "results/temporal_model/threshold_results.csv"
)


def load_predictions():
    """Load the test predictions generated during evaluation."""

    print("========================================")
    print(" THRESHOLD TUNING")
    print("========================================")

    print("Loading predictions...")

    data = pd.read_csv(PREDICTIONS_PATH)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def identify_columns(data):
    """Identify target and probability columns."""

    print()
    print("========== IDENTIFYING COLUMNS ==========")

    print("Available columns:")

    for column in data.columns:
        print(f"  {column}")

    target_candidates = [
        "target_24h",
        "y_true",
        "target",
        "actual",
    ]

    probability_candidates = [
        "probability",
        "predicted_probability",
        "prediction_probability",
        "y_probability",
        "prob",
    ]

    probability_column = "flare_probability"
    target_column = "target_24h"

    for column in target_candidates:
        if column in data.columns:
            target_column = column
            break

    for column in probability_candidates:
        if column in data.columns:
            probability_column = column
            break

    if target_column is None:
        raise ValueError(
            "Could not find target column."
        )

    if probability_column is None:
        raise ValueError(
            "Could not find probability column."
        )

    print(
        f"Target column: {target_column}"
    )

    print(
        f"Probability column: {probability_column}"
    )

    return target_column, probability_column


def evaluate_thresholds(
    y_true,
    probabilities,
):
    """Evaluate model performance at different thresholds."""

    print()
    print("========== TESTING THRESHOLDS ==========")

    thresholds = np.arange(
        0.01,
        1.00,
        0.01,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )

    results = pd.DataFrame(results)

    return results


def display_best_thresholds(results):
    """Display the best thresholds according to F1 and recall."""

    print()
    print("========== BEST THRESHOLDS ==========")

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    best_recall = results.loc[
        results["recall"].idxmax()
    ]

    print()
    print("BEST F1 THRESHOLD")
    print("------------------")

    print(
        f"Threshold : "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1['f1']:.4f}"
    )

    print()
    print("BEST RECALL THRESHOLD")
    print("----------------------")

    print(
        f"Threshold : "
        f"{best_recall['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_recall['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_recall['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_recall['f1']:.4f}"
    )


def save_results(results):
    """Save threshold evaluation results."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========== SAVING ==========")

    print(
        f"Saved: {OUTPUT_PATH}"
    )


def main():

    data = load_predictions()

    target_column, probability_column = (
        identify_columns(data)
    )

    y_true = data[
        target_column
    ].astype(int).to_numpy()

    probabilities = data[
        probability_column
    ].to_numpy()

    results = evaluate_thresholds(
        y_true,
        probabilities,
    )

    display_best_thresholds(
        results
    )

    save_results(
        results
    )

    print()
    print("========================================")
    print(" THRESHOLD TUNING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()