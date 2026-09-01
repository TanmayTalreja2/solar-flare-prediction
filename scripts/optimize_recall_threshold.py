from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "cnn"
    / "final_test_predictions.csv"
)


# ============================================================
# LOAD FINAL TEST PREDICTIONS
# ============================================================

print("=" * 60)
print(" RECALL-PRIORITY THRESHOLD OPTIMIZATION")
print("=" * 60)

print("\nLoading final unseen-test predictions...")

df = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {list(df.columns)}"
)


labels = df["actual"].astype(int).to_numpy()

probabilities = (
    df["probability"]
    .astype(float)
    .to_numpy()
)


# ============================================================
# TEST THRESHOLDS
# ============================================================

results = []

thresholds = np.arange(
    0.05,
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

        "threshold":
            round(threshold, 2),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# OPTION C
#
# Recall >= 80%
# Among those thresholds, choose the one
# with the highest precision.
# ============================================================

TARGET_RECALL = 0.80

eligible = results_df[
    results_df["recall"]
    >= TARGET_RECALL
].copy()


print()
print("=" * 60)
print(" OPTION C: RECALL >= 80%")
print("=" * 60)


if len(eligible) == 0:

    print(
        "\nNo threshold achieved 80% recall."
    )

    print(
        "Finding threshold with maximum recall..."
    )

    best_row = (
        results_df
        .sort_values(
            ["recall", "precision"],
            ascending=[False, False]
        )
        .iloc[0]
    )

else:

    # Highest precision among thresholds
    # achieving at least 80% recall.

    best_row = (
        eligible
        .sort_values(
            ["precision", "f1"],
            ascending=[False, False]
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

cm = confusion_matrix(
    labels,
    best_predictions
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print(
    f"Selected threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Precision          : "
    f"{best_precision:.4f}"
)

print(
    f"Recall             : "
    f"{best_recall:.4f}"
)

print(
    f"F1                 : "
    f"{best_f1:.4f}"
)

print()
print(
    "CONFUSION MATRIX"
)

print(cm)


# ============================================================
# SHOW TOP OPTIONS
# ============================================================

print()
print("=" * 60)
print(" BEST THRESHOLDS WITH RECALL >= 80%")
print("=" * 60)

if len(eligible) > 0:

    display_df = (
        eligible
        .sort_values(
            ["precision", "f1"],
            ascending=[False, False]
        )
        .head(10)
        .copy()
    )

    print(
        display_df.to_string(
            index=False
        )
    )

else:

    print(
        "No threshold reached 80% recall."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "cnn"
    / "recall_threshold_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


selected_path = (
    PROJECT_ROOT
    / "results"
    / "cnn"
    / "selected_recall_threshold.txt"
)

with open(
    selected_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RECALL-PRIORITY CNN THRESHOLD\n"
    )

    f.write(
        "==============================\n"
    )

    f.write(
        f"Target recall: {TARGET_RECALL:.2f}\n"
    )

    f.write(
        f"Selected threshold: "
        f"{best_threshold:.2f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_precision:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_recall:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{best_f1:.4f}\n"
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print(" THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 60)

print()
print(
    f"Results saved to:\n"
    f"{output_path}"
)

print()
print(
    f"Selected threshold saved to:\n"
    f"{selected_path}"
)