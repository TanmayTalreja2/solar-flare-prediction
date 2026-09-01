from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "cnn"
)


TRAINING_METRICS_PATH = (
    RESULTS_DIR
    / "training_metrics.csv"
)


TEST_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)


PLOTS_DIR = (
    RESULTS_DIR
    / "plots"
)


PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print()

print(
    "=" * 60
)

print(
    " GENERATING CNN RESULT VISUALIZATIONS"
)

print(
    "=" * 60
)


print()

print(
    "Loading training metrics..."
)


history = pd.read_csv(
    TRAINING_METRICS_PATH
)


print(
    f"Loaded {len(history)} training epochs."
)


print()

print(
    "Loading final test predictions..."
)


predictions = pd.read_csv(
    TEST_PREDICTIONS_PATH
)


print(
    f"Loaded {len(predictions)} test predictions."
)


# ============================================================
# SAFETY CHECK
# ============================================================

required_columns = [

    "actual",
    "probability",
    "prediction",

]


missing_columns = [

    column

    for column in required_columns

    if column not in predictions.columns

]


if missing_columns:

    raise ValueError(

        "Missing columns in "
        f"final_test_predictions.csv: "
        f"{missing_columns}"

    )


actuals = predictions[
    "actual"
].astype(
    int
).values


probabilities = predictions[
    "probability"
].astype(
    float
).values


predicted_classes = predictions[
    "prediction"
].astype(
    int
).values


# ============================================================
# PLOT 1
# TRAINING AND VALIDATION LOSS
# ============================================================

print()

print(
    "Creating training loss plot..."
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history["epoch"],

    history["train_loss"],

    marker="o",

    label="Training Loss",

)


plt.plot(

    history["epoch"],

    history["validation_loss"],

    marker="o",

    label="Validation Loss",

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "CNN Training and Validation Loss"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3,
)


plt.tight_layout()


plt.savefig(

    PLOTS_DIR
    / "training_validation_loss.png",

    dpi=300,

)


plt.close()


# ============================================================
# PLOT 2
# VALIDATION ROC-AUC AND PR-AUC
# ============================================================

print(
    "Creating validation metrics plot..."
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history["epoch"],

    history["validation_roc_auc"],

    marker="o",

    label="Validation ROC-AUC",

)


plt.plot(

    history["epoch"],

    history["validation_pr_auc"],

    marker="o",

    label="Validation PR-AUC",

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Score"
)


plt.title(
    "CNN Validation Performance"
)


plt.ylim(
    0,
    1,
)


plt.legend()


plt.grid(
    True,
    alpha=0.3,
)


plt.tight_layout()


plt.savefig(

    PLOTS_DIR
    / "validation_metrics.png",

    dpi=300,

)


plt.close()


# ============================================================
# PLOT 3
# CONFUSION MATRIX
# ============================================================

print(
    "Creating confusion matrix..."
)


matrix = confusion_matrix(

    actuals,

    predicted_classes,

)


display = ConfusionMatrixDisplay(

    confusion_matrix=matrix,

    display_labels=[

        "No Flare",

        "Flare",

    ],

)


figure, axis = plt.subplots(
    figsize=(6, 5)
)


display.plot(
    ax=axis
)


plt.title(
    "CNN Confusion Matrix - Final Test Set"
)


plt.tight_layout()


plt.savefig(

    PLOTS_DIR
    / "confusion_matrix.png",

    dpi=300,

)


plt.close()


# ============================================================
# PLOT 4
# ROC CURVE
# ============================================================

print(
    "Creating ROC curve..."
)


false_positive_rate, true_positive_rate, _ = roc_curve(

    actuals,

    probabilities,

)


roc_auc = auc(

    false_positive_rate,

    true_positive_rate,

)


plt.figure(
    figsize=(7, 6)
)


plt.plot(

    false_positive_rate,

    true_positive_rate,

    label=(
        f"ROC Curve "
        f"(AUC = {roc_auc:.4f})"
    ),

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve - Final Unseen Test Set"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3,
)


plt.tight_layout()


plt.savefig(

    PLOTS_DIR
    / "roc_curve.png",

    dpi=300,

)


plt.close()


# ============================================================
# PLOT 5
# PRECISION-RECALL CURVE
# ============================================================

print(
    "Creating Precision-Recall curve..."
)


precision, recall, _ = precision_recall_curve(

    actuals,

    probabilities,

)


pr_auc = average_precision_score(

    actuals,

    probabilities,

)


plt.figure(
    figsize=(7, 6)
)


plt.plot(

    recall,

    precision,

    label=(
        f"PR Curve "
        f"(AP = {pr_auc:.4f})"
    ),

)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve - Final Unseen Test Set"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3,
)


plt.tight_layout()


plt.savefig(

    PLOTS_DIR
    / "precision_recall_curve.png",

    dpi=300,

)


plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print(
    "=" * 60
)

print(
    " VISUALIZATIONS COMPLETE"
)

print(
    "=" * 60
)


print()

print(
    "Generated plots:"
)


for plot_file in [

    "training_validation_loss.png",

    "validation_metrics.png",

    "confusion_matrix.png",

    "roc_curve.png",

    "precision_recall_curve.png",

]:

    print(

        PLOTS_DIR
        / plot_file

    )


print()

print(
    "Final Test Performance:"
)


print(
    f"ROC-AUC : {roc_auc:.4f}"
)


print(
    f"PR-AUC  : {pr_auc:.4f}"
)


print()

print(
    "Done."
)