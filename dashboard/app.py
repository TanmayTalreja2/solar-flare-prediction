import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.predict import SolarFlarePredictor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Flare Prediction Dashboard",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_predictor():
    return SolarFlarePredictor(
        str(project_root / "models" /
            "xgboost_2012_temporal_features.joblib")
    )


@st.cache_data
def load_data():

    data_path = (
        project_root /
        "data" /
        "processed" /
        "features" /
        "sharp_goes_temporal_features_2012_full.parquet"
    )

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()

    df = pd.read_parquet(data_path)

    df["observation_time"] = pd.to_datetime(
        df["observation_time"]
    )

    return df


predictor = load_predictor()
data = load_data()


# ============================================================
# TEST PERIOD
# ============================================================

test_start = pd.Timestamp("2012-07-01 00:00:00")

test_data = data[
    data["observation_time"] >= test_start
].copy()

active_regions = test_data["NOAA_AR"].unique()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("☀️ Solar Flare Prediction")

st.sidebar.markdown(
    """
    **24-Hour Solar Flare Early Warning System**

    Uses SHARP magnetic-field observations,
    GOES observations and temporal features
    to estimate flare risk.
    """
)

selected_ar = st.sidebar.selectbox(
    "Select Active Region (NOAA AR)",
    sorted(active_regions)
)

ar_data = (
    test_data[test_data["NOAA_AR"] == selected_ar]
    .sort_values("observation_time")
    .reset_index(drop=True)
)

timestamps = (
    ar_data["observation_time"]
    .dt.strftime("%Y-%m-%d %H:%M:%S")
    .tolist()
)

selected_time = st.sidebar.selectbox(
    "Select Observation Time",
    timestamps,
    index=len(timestamps) - 1
)


# ============================================================
# SELECT OBSERVATION
# ============================================================

idx = timestamps.index(selected_time)

observation = ar_data.iloc[[idx]].copy()

prediction_result = predictor.predict(observation)

flare_prob = float(
    prediction_result.iloc[0]["flare_probability"]
)

risk_level = prediction_result.iloc[0]["risk_level"]

actual_target = observation.iloc[0].get(
    "target_24h",
    None
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def readable_feature_name(name):

    names = {

        "USFLUX":
            "Total Unsigned Magnetic Flux",

        "TOTUSJH":
            "Total Current Helicity",

        "TOTUSJH_CHANGE_6h":
            "Total Current Helicity Change (6h)",

        "TOTUSJH_CHANGE_12h":
            "Total Current Helicity Change (12h)",

        "TOTPOT":
            "Total Magnetic Free Energy Proxy",

        "MEANSHR":
            "Mean Shear Angle",

        "MEANGAM":
            "Mean Magnetic Field Angle",

        "MEANGBT":
            "Mean Gradient of Magnetic Field",

        "TOTUSJZ":
            "Total Vertical Current",

    }

    return names.get(
        name,
        name.replace("_", " ").title()
    )


def format_value(value):

    try:
        value = float(value)
    except:
        return str(value)

    if abs(value) >= 1e6:
        return f"{value:.3e}"

    return f"{value:.3f}"


# ============================================================
# MAIN HEADER
# ============================================================

st.title("☀️ Solar Flare Early Warning System")

st.markdown(
    """
    ### AI-assisted prediction of solar flare risk

    The system analyzes magnetic and temporal properties
    of solar active regions to estimate the probability
    of an M/X-class flare within the next 24 hours.
    """
)

st.markdown("---")


# ============================================================
# OBSERVATION INFO
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Active Region",
        f"NOAA {selected_ar}"
    )

with col2:
    st.metric(
        "Observation Time",
        selected_time
    )

with col3:
    st.metric(
        "Prediction Horizon",
        "24 Hours"
    )


# ============================================================
# PREDICTION STATUS
# ============================================================

st.markdown("---")

st.header("🚨 Prediction Status")

c1, c2 = st.columns([1, 2])

with c1:

    st.markdown(
        f"""
        ## {flare_prob * 100:.2f}%

        **Predicted Flare Probability**
        """
    )

    st.markdown(
        f"""
        ### Risk Level

        <span style="
        color:{get_risk_color(risk_level)};
        font-size:28px;
        font-weight:bold;">
        {risk_level}
        </span>
        """,
        unsafe_allow_html=True
    )

    if flare_prob >= 0.01:

        st.success(
            "⚠️ FLARE RISK DETECTED"
        )

    else:

        st.info(
            "No significant flare risk detected."
        )

    if actual_target is not None:

        if actual_target == 1:
            st.write(
                "📌 **Observed outcome:** Flare occurred"
            )
        else:
            st.write(
                "📌 **Observed outcome:** No flare"
            )


with c2:

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=flare_prob * 100,

            title={
                "text": "Predicted Flare Risk (%)"
            },

            gauge={

                "axis": {
                    "range": [0, 100]
                },

                "steps": [

                    {
                        "range": [0, 1],
                        "color": "green"
                    },

                    {
                        "range": [1, 5],
                        "color": "orange"
                    },

                    {
                        "range": [5, 20],
                        "color": "red"
                    },

                    {
                        "range": [20, 100],
                        "color": "darkred"
                    }

                ],

                "bar": {
                    "color": "black"
                }

            }
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# HUMAN-READABLE INTERPRETATION
# ============================================================

st.markdown("---")

st.header("🧠 AI Interpretation")

if risk_level == "VERY HIGH RISK":

    st.warning(
        "The model indicates a very high predicted risk "
        "of an M/X-class solar flare within the next "
        "24 hours. This active region warrants close monitoring."
    )

elif risk_level == "HIGH RISK":

    st.warning(
        "The model indicates an elevated flare risk. "
        "Continued monitoring is recommended."
    )

elif risk_level == "MODERATE RISK":

    st.info(
        "The model indicates a moderate flare risk."
    )

else:

    st.success(
        "The model indicates a relatively low flare risk."
    )

st.caption(
    "The displayed probability is the model's predicted "
    "probability and should not be interpreted as a guaranteed "
    "physical probability of a flare."
)
# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.header("📊 Model Performance")

st.markdown(
    """
    Performance evaluated on the held-out 2012 test period.
    The models use different information sources:
    **XGBoost** uses SHARP/GOES temporal features, while
    **CNN** analyzes magnetogram images.
    """
)

performance_df = pd.DataFrame({
    "Model": [
        "XGBoost",
        "CNN",
        "50/50 Ensemble"
    ],
    "ROC-AUC": [
        0.7408,
        0.7218,
        0.7489
    ],
    "PR-AUC": [
        0.6955,
        0.7866,
        0.7667
    ]
})

st.dataframe(
    performance_df.style.format({
        "ROC-AUC": "{:.4f}",
        "PR-AUC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Performance chart
metric = st.radio(
    "Select evaluation metric:",
    ["ROC-AUC", "PR-AUC"],
    horizontal=True
)

fig_performance = px.bar(
    performance_df,
    x="Model",
    y=metric,
    text=metric,
    title=f"{metric} Comparison"
)

fig_performance.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_performance.update_yaxes(
    range=[0, 1]
)

st.plotly_chart(
    fig_performance,
    use_container_width=True
)

# Best model explanation
if metric == "PR-AUC":
    st.success(
        "🏆 Best PR-AUC: CNN — 0.7866. "
        "The CNN performs particularly well on the held-out "
        "magnetogram evaluation subset."
    )
else:
    st.success(
        "🏆 Best ROC-AUC: 50/50 Ensemble — 0.7489. "
        "Combining the two model outputs provides the strongest "
        "ROC-AUC among the evaluated configurations."
    )

# ============================================================
# KEY MAGNETIC INDICATORS
# ============================================================

st.markdown("---")

st.header("🧲 Key Magnetic Indicators")

importance = predictor.model.feature_importances_
features = predictor.features

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Remove identifiers / non-physical identifiers
excluded = {
    "HARPNUM",
    "NOAA_AR",
    "observation_time",
    "target_24h"
}

importance_df = importance_df[
    ~importance_df["Feature"].isin(excluded)
]

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(5)
)

for _, row in importance_df.iterrows():

    feature = row["Feature"]

    value = observation.iloc[0].get(
        feature,
        None
    )

    if value is None:
        continue

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.write(
            f"**{readable_feature_name(feature)}**"
        )

    with col2:
        st.write(
            f"`{format_value(value)}`"
        )

    with col3:
        st.write(
            f"Importance: `{row['Importance']:.4f}`"
        )


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

st.subheader("Top Model Features")

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 5 Global Model Features"
)

fig_imp.update_layout(
    yaxis={
        "categoryorder": "total ascending"
    }
)

st.plotly_chart(
    fig_imp,
    use_container_width=True
)


# ============================================================
# TEMPORAL EVOLUTION
# ============================================================

st.markdown("---")

st.header(
    f"📈 Temporal Evolution: AR {selected_ar}"
)

st.markdown(
    "Historical magnetic parameters up to the selected observation."
)

historical_data = ar_data.iloc[:idx + 1].copy()

plot_columns = [
    "TOTUSJH",
    "TOTPOT",
    "USFLUX"
]

plot_columns = [
    c for c in plot_columns
    if c in historical_data.columns
]

hist_plot_data = historical_data[
    ["observation_time"] + plot_columns
].copy()

for col in plot_columns:

    std = hist_plot_data[col].std()

    if std != 0:

        hist_plot_data[col] = (
            hist_plot_data[col]
            - hist_plot_data[col].mean()
        ) / std

fig_temporal = px.line(
    hist_plot_data,
    x="observation_time",
    y=plot_columns,
    title="Normalized Evolution of Magnetic Features"
)

st.plotly_chart(
    fig_temporal,
    use_container_width=True
)


# ============================================================
# RAW TEMPORAL SIGNALS
# ============================================================

st.markdown("---")

st.header("🔬 Temporal Signal Values")

cols_to_show = [
    "TOTUSJH_CHANGE_6h",
    "TOTUSJH_CHANGE_12h",
    "TOTPOT_RELCHANGE_1h",
    "TOTUSJH_ROLLSTD_12h"
]

cols_to_show = [
    c for c in cols_to_show
    if c in observation.columns
]

if cols_to_show:

    st.dataframe(
        observation[cols_to_show],
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Solar Flare Prediction System | "
    "XGBoost + CNN | SHARP/GOES + Magnetogram Analysis | "
    "24-hour prediction window"
)