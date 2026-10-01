from pathlib import Path
import sys

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.predict import SolarFlarePredictor

TEST_START = pd.Timestamp("2012-07-01 00:00:00")

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_2012_temporal_features_25params.joblib"
FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features" /
    "sharp_goes_temporal_features_2012_full_25params.parquet"
)
CNN_METRICS_PATH = PROJECT_ROOT / "results" / "cnn" / "final_metrics.csv"
CNN_CONFUSION_IMAGE = PROJECT_ROOT / "results" / "cnn" / "plots" / "confusion_matrix.png"
FUSION_METRICS_PATH = PROJECT_ROOT / "results" / "ensemble_25params" / "fusion_results.csv"
FUSION_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "ensemble_25params" / "fusion_predictions.csv"


@st.cache_resource
def load_predictor():
    return SolarFlarePredictor(str(MODEL_PATH))


@st.cache_data
def load_data():

    if not FEATURES_PATH.exists():
        st.error(f"Data file not found: {FEATURES_PATH}")
        st.stop()

    df = pd.read_parquet(FEATURES_PATH)
    df["observation_time"] = pd.to_datetime(df["observation_time"])

    return df


@st.cache_data
def load_test_data():

    df = load_data()

    return df[df["observation_time"] >= TEST_START].copy()


@st.cache_data
def load_performance_metrics():

    predictor = load_predictor()

    rows = []

    xgb_metrics = predictor.model_package.get("metrics", {})

    rows.append({
        "Model": "XGBoost (17 params)",
        "ROC-AUC": xgb_metrics.get("roc_auc"),
        "PR-AUC": xgb_metrics.get("pr_auc"),
    })

    if CNN_METRICS_PATH.exists():

        cnn_df = pd.read_csv(CNN_METRICS_PATH)

        rows.append({
            "Model": "CNN",
            "ROC-AUC": cnn_df["test_roc_auc"].iloc[0],
            "PR-AUC": cnn_df["test_pr_auc"].iloc[0],
        })

    if FUSION_METRICS_PATH.exists():

        fusion_df = pd.read_csv(FUSION_METRICS_PATH)
        fusion_row = fusion_df[fusion_df["model"] == "Fusion"]

        if not fusion_row.empty:

            rows.append({
                "Model": "Fusion",
                "ROC-AUC": fusion_row["roc_auc"].iloc[0],
                "PR-AUC": fusion_row["pr_auc"].iloc[0],
            })

    return pd.DataFrame(rows)


@st.cache_data
def compute_xgboost_test_confusion():
    """Recompute XGBoost's confusion matrix live on the held-out test
    period, using the same predictor the rest of the app uses. Nothing
    here is hardcoded -- it reruns actual inference on real test rows."""

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    predictor = load_predictor()
    test_data = load_test_data()

    results = predictor.predict(test_data)

    y_true = test_data["target_24h"].astype(int).values
    y_pred = results["prediction"].values

    cm = confusion_matrix(y_true, y_pred)

    return {
        "confusion_matrix": cm,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def readable_feature_name(name):

    names = {
        "USFLUX": "Total Unsigned Magnetic Flux",
        "TOTUSJH": "Total Current Helicity",
        "TOTUSJH_CHANGE_6h": "Total Current Helicity Change (6h)",
        "TOTUSJH_CHANGE_12h": "Total Current Helicity Change (12h)",
        "TOTPOT": "Total Magnetic Free Energy Proxy",
        "MEANSHR": "Mean Shear Angle",
        "MEANGAM": "Mean Magnetic Field Angle",
        "MEANGBT": "Mean Gradient of Magnetic Field",
        "TOTUSJZ": "Total Vertical Current",
        "R_VALUE": "Polarity Inversion Line Complexity (R_VALUE)",
        "ABSNJZH": "Absolute Net Current Helicity",
        "SAVNCPP": "Sum of Absolute Vertical Current near PIL",
        "SHRGT45": "Area with Shear Angle > 45deg",
    }

    return names.get(name, name.replace("_", " ").title())


def format_value(value):

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(value) >= 1e6:
        return f"{value:.3e}"

    return f"{value:.3f}"


def get_risk_color(risk):

    if risk == "LOW RISK":
        return "green"

    if risk == "MODERATE RISK":
        return "orange"

    if risk == "HIGH RISK":
        return "red"

    if risk == "VERY HIGH RISK":
        return "darkred"

    return "gray"
