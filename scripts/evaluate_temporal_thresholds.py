from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


MODEL_PATH = Path(
    "models/xgboost_2012_temporal_features.joblib"
)

DATA_PATH = Path(
    "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)


ORIGINAL_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
    "MEANSHR",
]


TEMPORAL_FEATURES = [
    column
    for column in pd.read_parquet(DATA_PATH, columns=None).columns
    if (
        "_CHANGE_" in column
        or "_RELCHANGE_" in column
        or "_ROLLSTD_" in column
    )
]


def main():

    print("========================================")
    print(" TEMPORAL THRESHOLD OPTIMIZATION")
    print("========================================")

    print("Loading dataset...")

    data = pd.read_parquet(DATA_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"]
    )

    data = data.sort_values(
        "observation_time"
    ).reset_index(drop=True)

    cutoff = pd.Timestamp("2012-07-01")

    train = data[
        data["observation_time"] < cutoff
    ].copy()

    test = data[
        data["observation_time"] >= cutoff
    ].copy()

    print(f"Training rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")

    feature_columns = [
        column
        for column in data.columns
        if column not in [
            "observation_time",
            "NOAA_AR",
            "flare_class",
            "target_24h",
        ]
    ]

    print()
    print(
        f"Features: {len(feature_columns)}"
    )

    X_test = test[feature_columns]
    y_test = test["target_24h"]

    print()
    print("Loading model...")

    saved = joblib.load(MODEL_PATH)

    # Support either a pipeline or a dictionary-style saved model
    if isinstance(saved, dict):

        model = saved["model"]
        imputer = saved.get("imputer")

    else:

        model = saved
        imputer = None

    print("Model loaded.")

    # --------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------

    if imputer is not None:

        X_test = imputer.transform(
            X_test
        )

    # --------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print("========== BASE METRICS ==========")

    print(
        f"ROC-AUC : "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"PR-AUC  : "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    # --------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------

    print()
    print("========== THRESHOLD SEARCH ==========")

    thresholds = np.arange(
        0.05,
        0.51,
        0.01,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive": predictions.sum(),
            }
        )

    results = pd.DataFrame(results)

    print(
        results.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------
    # BEST F1
    # --------------------------------------------------

    best = results.loc[
        results["f1"].idxmax()
    ]

    print()
    print("========== BEST F1 THRESHOLD ==========")

    print(
        f"Threshold          : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision          : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1                 : "
        f"{best['f1']:.4f}"
    )

    print(
        f"Predicted positives: "
        f"{int(best['predicted_positive'])}"
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    best_threshold = best["threshold"]

    best_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        best_predictions,
    )

    print()
    print("========== CONFUSION MATRIX ==========")

    print(cm)

    print()
    print("========================================")
    print(" THRESHOLD OPTIMIZATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()