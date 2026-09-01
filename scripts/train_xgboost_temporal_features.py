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


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)

MODEL_PATH = Path(
    "models/"
    "xgboost_2012_temporal_features.joblib"
)

RESULTS_DIR = Path("results/xgboost")


# ============================================================
# FEATURES
# ============================================================

ORIGINAL_FEATURES = [
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

BASE_TEMPORAL_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]

TEMPORAL_WINDOWS = [
    "1h",
    "3h",
    "6h",
    "12h",
]

TEMPORAL_OPERATIONS = [
    "CHANGE",
    "RELCHANGE",
    "ROLLSTD",
]

TARGET_COLUMN = "target_24h"

RANDOM_STATE = 42

# Final test begins here.
# July-December remains completely unseen until final evaluation.
TEST_START = pd.Timestamp("2012-07-01 00:00:00")

# Validation is the last part of the training period.
VALIDATION_START = pd.Timestamp("2012-05-01 00:00:00")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print(" IMPROVED TEMPORAL XGBOOST")
    print("=" * 60)

    print("Loading temporal feature dataset...")

    data = pd.read_parquet(INPUT_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


# ============================================================
# BUILD FEATURE LIST
# ============================================================

def get_temporal_features():

    temporal_features = []

    for feature in BASE_TEMPORAL_FEATURES:

        for operation in TEMPORAL_OPERATIONS:

            for window in TEMPORAL_WINDOWS:

                temporal_features.append(
                    f"{feature}_{operation}_{window}"
                )

    return temporal_features


def prepare_features(data):

    temporal_features = get_temporal_features()

    feature_columns = (
        ORIGINAL_FEATURES
        + temporal_features
    )

    missing = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    print()
    print("========== FEATURE VALIDATION ==========")
    print(f"Original features : {len(ORIGINAL_FEATURES)}")
    print(f"Temporal features : {len(temporal_features)}")
    print(f"Total features    : {len(feature_columns)}")

    if missing:

        print("\nMissing features:")

        for feature in missing:
            print(" ", feature)

        raise ValueError(
            "Required features are missing."
        )

    print("All required features found.")

    return feature_columns


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(data, feature_columns):

    print()
    print("========== CHRONOLOGICAL SPLIT ==========")

    data = data.sort_values(
        "observation_time"
    ).reset_index(drop=True)

    train_mask = (
        data["observation_time"]
        < VALIDATION_START
    )

    validation_mask = (
        (data["observation_time"] >= VALIDATION_START)
        & (data["observation_time"] < TEST_START)
    )

    test_mask = (
        data["observation_time"] >= TEST_START
    )

    train_data = data.loc[
        train_mask
    ].copy()

    validation_data = data.loc[
        validation_mask
    ].copy()

    test_data = data.loc[
        test_mask
    ].copy()

    print(
        f"Training rows   : {len(train_data)}"
    )

    print(
        f"Validation rows : {len(validation_data)}"
    )

    print(
        f"Test rows       : {len(test_data)}"
    )

    print()
    print("TRAIN:")
    print(
        train_data["observation_time"].min(),
        "->",
        train_data["observation_time"].max(),
    )

    print()
    print("VALIDATION:")
    print(
        validation_data["observation_time"].min(),
        "->",
        validation_data["observation_time"].max(),
    )

    print()
    print("FINAL TEST:")
    print(
        test_data["observation_time"].min(),
        "->",
        test_data["observation_time"].max(),
    )

    X_train = train_data[feature_columns].copy()
    X_validation = validation_data[feature_columns].copy()
    X_test = test_data[feature_columns].copy()

    y_train = train_data[TARGET_COLUMN].astype(int)
    y_validation = validation_data[TARGET_COLUMN].astype(int)
    y_test = test_data[TARGET_COLUMN].astype(int)

    print()
    print("TRAIN TARGET:")
    print(y_train.value_counts().sort_index())

    print()
    print("VALIDATION TARGET:")
    print(y_validation.value_counts().sort_index())

    print()
    print("TEST TARGET:")
    print(y_test.value_counts().sort_index())

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        train_data,
        validation_data,
        test_data,
    )


# ============================================================
# IMPUTATION
# ============================================================

def prepare_features_data(
    X_train,
    X_validation,
    X_test,
):

    print()
    print("========== MEDIAN IMPUTATION ==========")

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_validation_imp = imputer.transform(
        X_validation
    )

    X_test_imp = imputer.transform(
        X_test
    )

    print(
        "Imputer fitted ONLY on training data."
    )

    return (
        X_train_imp,
        X_validation_imp,
        X_test_imp,
        imputer,
    )


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

def get_model_configs(scale_pos_weight):

    return [

        {
            "name": "balanced_depth4",
            "n_estimators": 1000,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.0,
        },

        {
            "name": "balanced_depth5",
            "n_estimators": 1000,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.0,
        },

        {
            "name": "regularized_depth4",
            "n_estimators": 1200,
            "max_depth": 4,
            "learning_rate": 0.025,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.1,
        },

        {
            "name": "regularized_depth5",
            "n_estimators": 1200,
            "max_depth": 5,
            "learning_rate": 0.025,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.1,
        },

    ]


# ============================================================
# TRAIN + VALIDATE
# ============================================================

def train_and_select_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())

    scale_pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    print()
    print("========== CLASS BALANCE ==========")
    print(f"Negative samples : {negative}")
    print(f"Positive samples : {positive}")
    print(
        f"scale_pos_weight : "
        f"{scale_pos_weight:.3f}"
    )

    configs = get_model_configs(
        scale_pos_weight
    )

    results = []

    best_model = None
    best_config = None
    best_pr_auc = -np.inf

    for config in configs:

        print()
        print("=" * 60)
        print(
            f"TESTING CONFIGURATION: "
            f"{config['name']}"
        )
        print("=" * 60)

        model = XGBClassifier(

            n_estimators=config["n_estimators"],

            max_depth=config["max_depth"],

            learning_rate=config["learning_rate"],

            min_child_weight=config[
                "min_child_weight"
            ],

            subsample=config["subsample"],

            colsample_bytree=config[
                "colsample_bytree"
            ],

            gamma=config["gamma"],

            objective="binary:logistic",

            eval_metric="aucpr",

            scale_pos_weight=scale_pos_weight,

            random_state=RANDOM_STATE,

            n_jobs=-1,

        )

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],

            verbose=False,
        )

        validation_prob = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_roc = roc_auc_score(
            y_validation,
            validation_prob,
        )

        validation_pr = (
            average_precision_score(
                y_validation,
                validation_prob,
            )
        )

        results.append({

            "model": config["name"],

            "validation_roc_auc":
                validation_roc,

            "validation_pr_auc":
                validation_pr,

        })

        print(
            f"Validation ROC-AUC : "
            f"{validation_roc:.4f}"
        )

        print(
            f"Validation PR-AUC  : "
            f"{validation_pr:.4f}"
        )

        if validation_pr > best_pr_auc:

            best_pr_auc = validation_pr
            best_model = model
            best_config = config

            print(
                ">>> NEW BEST MODEL"
            )

    results_df = pd.DataFrame(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_DIR
        / "hyperparameter_results.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print(" BEST CONFIGURATION")
    print("=" * 60)

    print(
        best_config["name"]
    )

    print(
        f"Validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )

    return (
        best_model,
        best_config,
        results_df,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    model,
    X_validation,
    y_validation,
):

    print()
    print("========== THRESHOLD OPTIMIZATION ==========")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    thresholds = np.arange(
        0.01,
        0.51,
        0.01,
    )

    best_threshold = 0.5
    best_f1 = -1

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        threshold_results.append({

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1": f1,

        })

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    threshold_df = pd.DataFrame(
        threshold_results
    )

    threshold_df.to_csv(
        RESULTS_DIR
        / "threshold_results.csv",
        index=False,
    )

    print(
        f"Best threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1  : "
        f"{best_f1:.4f}"
    )

    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print(
        f"Precision       : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_row['recall']:.4f}"
    )

    return best_threshold


# ============================================================
# FINAL TEST
# ============================================================

def evaluate_final_model(
    model,
    X_test,
    y_test,
    threshold,
):

    print()
    print("=" * 60)
    print(" FINAL UNSEEN TEST EVALUATION")
    print("=" * 60)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
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

    cm = confusion_matrix(
        y_test,
        predictions,
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
    print("CONFUSION MATRIX")
    print(cm)

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return {

        "probabilities":
            probabilities,

        "predictions":
            predictions,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "confusion_matrix":
            cm,

    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    feature_columns,
):

    print()
    print(
        "========== TOP FEATURES =========="
    )

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(
        ascending=False
    )

    print(
        importance.head(20)
    )

    importance.to_csv(
        RESULTS_DIR
        / "feature_importance.csv"
    )

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    imputer,
    feature_columns,
    metrics,
    threshold,
    best_config,
):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    package = {

        "model":
            model,

        "imputer":
            imputer,

        "features":
            feature_columns,

        "target":
            TARGET_COLUMN,

        "threshold":
            threshold,

        "metrics": {

            "roc_auc":
                metrics["roc_auc"],

            "pr_auc":
                metrics["pr_auc"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

        },

        "best_config":
            best_config,

        "training_period":
            "2012-01-01 to 2012-04-30",

        "validation_period":
            "2012-05-01 to 2012-06-30",

        "testing_period":
            "2012-07-01 to 2012-12-31",

        "random_state":
            RANDOM_STATE,

    }

    joblib.dump(
        package,
        MODEL_PATH,
    )

    print()
    print(
        "Model saved:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    data = load_data()

    feature_columns = prepare_features(
        data
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        train_data,
        validation_data,
        test_data,
    ) = chronological_split(
        data,
        feature_columns,
    )

    (
        X_train,
        X_validation,
        X_test,
        imputer,
    ) = prepare_features_data(
        X_train,
        X_validation,
        X_test,
    )

    (
        model,
        best_config,
        comparison,
    ) = train_and_select_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    threshold = find_best_threshold(
        model,
        X_validation,
        y_validation,
    )

    metrics = evaluate_final_model(
        model,
        X_test,
        y_test,
        threshold,
    )

    show_feature_importance(
        model,
        feature_columns,
    )

    save_model(
        model,
        imputer,
        feature_columns,
        metrics,
        threshold,
        best_config,
    )

    print()
    print("=" * 60)
    print(" IMPROVED XGBOOST TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Final ROC-AUC : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Final PR-AUC  : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Final F1      : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Threshold     : "
        f"{threshold:.2f}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()