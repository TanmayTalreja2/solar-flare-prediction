from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)

MODEL_PATH = Path(
    "models/"
    "xgboost_2012_temporal_features.joblib"
)

OUTPUT_DIR = Path("results/temporal_model")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained XGBoost model package.

    The saved package contains:
        - trained model
        - imputer
        - feature list
    """

    print("========== LOADING MODEL ==========")

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    imputer = package["imputer"]
    feature_columns = package["features"]

    print(f"Features expected by model: {len(feature_columns)}")

    return model, imputer, feature_columns


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the temporal feature dataset.
    """

    print()
    print("========== LOADING DATA ==========")

    data = pd.read_parquet(DATA_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data = data.sort_values(
        "observation_time"
    ).reset_index(drop=True)

    print(f"Rows: {len(data)}")

    return data


# ============================================================
# CHRONOLOGICAL TEST SPLIT
# ============================================================

def create_test_data(data):
    """
    Select the second half of 2012 as the
    chronological test set.

    Training:
        January -> June

    Testing:
        July -> December
    """

    print()
    print("========== CREATING TEST SET ==========")

    test_mask = (
        data["observation_time"]
        >= pd.Timestamp("2012-07-01")
    )

    test_data = data.loc[
        test_mask
    ].copy()

    print(f"Test rows: {len(test_data)}")

    print(
        f"Test period: "
        f"{test_data['observation_time'].min()} "
        f"→ "
        f"{test_data['observation_time'].max()}"
    )

    return test_data


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    test_data,
    feature_columns,
    imputer,
):
    """
    Extract model features and apply the
    already-fitted training imputer.

    IMPORTANT:
    We do NOT fit the imputer again.
    """

    print()
    print("========== PREPARING FEATURES ==========")

    X_test = test_data[
        feature_columns
    ].copy()

    y_test = test_data[
        "target_24h"
    ].astype(int)

    X_test = imputer.transform(
        X_test
    )

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    return X_test, y_test


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_test,
):
    """
    Generate probability predictions
    and binary predictions.

    probability:
        Probability of flare within 24h.

    prediction:
        Binary classification using threshold 0.5.
    """

    print()
    print("========== GENERATING PREDICTIONS ==========")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return probabilities, predictions


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_test,
    probabilities,
    predictions,
):
    """
    Calculate the main classification metrics.
    """

    print()
    print("========== METRICS ==========")

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

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_test,
    predictions,
):
    """
    Create and save the confusion matrix plot.
    """

    print()
    print("========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(cm)

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Flare",
            "Flare",
        ],
    ).plot(
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix - XGBoost"
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# ROC CURVE
# ============================================================

def save_roc_curve(
    y_test,
    probabilities,
):
    """
    Generate and save the ROC curve.
    """

    print()
    print("========== ROC CURVE ==========")

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost (AUC = {auc:.4f})",
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
        "ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "roc_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

def save_pr_curve(
    y_test,
    probabilities,
):
    """
    Generate and save the Precision-Recall curve.

    PR-AUC is particularly useful here because
    flare observations are much rarer than
    non-flare observations.
    """

    print()
    print("========== PRECISION-RECALL CURVE ==========")

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities,
        )
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"XGBoost (AP = {pr_auc:.4f})",
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "precision_recall_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_columns,
):
    """
    Extract XGBoost feature importance and
    save the top 20 features.
    """

    print()
    print("========== FEATURE IMPORTANCE ==========")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    )

    importance = (
        importance
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    print(importance)

    plt.figure(
        figsize=(9, 7)
    )

    importance.sort_values().plot(
        kind="barh"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 XGBoost Features"
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "feature_importance.png"
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    importance.to_csv(
        OUTPUT_DIR
        / "feature_importance.csv"
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test_data,
    probabilities,
    predictions,
):
    """
    Save every test observation with:

        observation time
        active region
        actual target
        predicted probability
        predicted class
    """

    print()
    print("========== SAVING PREDICTIONS ==========")

    output = test_data[
        [
            "observation_time",
            "NOAA_AR",
            "target_24h",
        ]
    ].copy()

    output[
        "flare_probability"
    ] = probabilities

    output[
        "prediction"
    ] = predictions

    path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):
    """
    Save evaluation metrics to CSV.
    """

    metrics_df = pd.DataFrame(
        [metrics]
    )

    path = (
        OUTPUT_DIR
        / "metrics.csv"
    )

    metrics_df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" TEMPORAL XGBOOST EVALUATION")
    print("========================================")

    model, imputer, feature_columns = (
        load_model()
    )

    data = load_data()

    test_data = create_test_data(
        data
    )

    X_test, y_test = prepare_features(
        test_data,
        feature_columns,
        imputer,
    )

    probabilities, predictions = (
        generate_predictions(
            model,
            X_test,
        )
    )

    metrics = calculate_metrics(
        y_test,
        probabilities,
        predictions,
    )

    save_confusion_matrix(
        y_test,
        predictions,
    )

    save_roc_curve(
        y_test,
        probabilities,
    )

    save_pr_curve(
        y_test,
        probabilities,
    )

    save_feature_importance(
        model,
        feature_columns,
    )

    save_predictions(
        test_data,
        probabilities,
        predictions,
    )

    save_metrics(
        metrics
    )

    print()
    print("========================================")
    print(" EVALUATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()