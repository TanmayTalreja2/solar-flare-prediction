from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).parent.parent

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
    / "dataset_labels.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "cnn"
    / "validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "cnn"
    / "validation_threshold_results.csv"
)


print("=" * 60)
print(" VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 60)

print("\nLoading validation predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


labels = df["actual"].astype(int).to_numpy()

probabilities = (
    df["probability"]
    .astype(float)
    .to_numpy()
)


# ============================================================
# SEARCH THRESHOLDS
# ============================================================

results = []

thresholds = np.arange(
    0.01,
    0.96,
    0.01
)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    results.append({

        "threshold": round(
            threshold,
            2
        ),

        "precision": precision,

        "recall": recall,

        "f1": f1,

    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# OPTION C
#
# Recall >= 80%
# Highest precision among eligible thresholds
# ============================================================

TARGET_RECALL = 0.80

eligible = results_df[
    results_df["recall"]
    >= TARGET_RECALL
].copy()


print()
print("=" * 60)
print(" OPTION C: VALIDATION RECALL >= 80%")
print("=" * 60)


if len(eligible) == 0:

    print(
        "\nNo threshold reaches 80% recall."
    )

    print(
        "Selecting threshold with highest validation recall."
    )

    best_row = (
        results_df
        .sort_values(
            ["recall", "precision"],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )

else:

    best_row = (
        eligible
        .sort_values(
            ["precision", "f1"],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )


best_threshold = float(
    best_row["threshold"]
)


best_predictions = (
    probabilities
    >= best_threshold
).astype(int)


best_precision = precision_score(
    labels,
    best_predictions,
    zero_division=0
)

best_recall = recall_score(
    labels,
    best_predictions,
    zero_division=0
)

best_f1 = f1_score(
    labels,
    best_predictions,
    zero_division=0
)


print()
print(
    f"Selected threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation precision: "
    f"{best_precision:.4f}"
)

print(
    f"Validation recall   : "
    f"{best_recall:.4f}"
)

print(
    f"Validation F1       : "
    f"{best_f1:.4f}"
)


# ============================================================
# SHOW ELIGIBLE THRESHOLDS
# ============================================================

print()
print("=" * 60)
print(" TOP VALIDATION THRESHOLDS")
print("=" * 60)

if len(eligible) > 0:

    print(
        eligible
        .sort_values(
            ["precision", "f1"],
            ascending=[
                False,
                False
            ]
        )
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


selected_path = (
    PROJECT_ROOT
    / "results"
    / "cnn"
    / "official_threshold.txt"
)

with open(
    selected_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "OFFICIAL CNN THRESHOLD\n"
    )

    f.write(
        "======================\n"
    )

    f.write(
        "Selected using validation data only.\n"
    )

    f.write(
        f"Target recall: {TARGET_RECALL:.2f}\n"
    )

    f.write(
        f"Threshold: {best_threshold:.2f}\n"
    )

    f.write(
        f"Validation precision: "
        f"{best_precision:.4f}\n"
    )

    f.write(
        f"Validation recall: "
        f"{best_recall:.4f}\n"
    )

    f.write(
        f"Validation F1: "
        f"{best_f1:.4f}\n"
    )


print()
print("=" * 60)
print(" COMPLETE")
print("=" * 60)

print()
print(
    f"Threshold results:\n{OUTPUT_PATH}"
)

print()
print(
    f"Official threshold:\n{selected_path}"
)