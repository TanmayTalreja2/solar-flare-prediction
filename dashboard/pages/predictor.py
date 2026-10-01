import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from common import (
    load_predictor,
    load_test_data,
    readable_feature_name,
    format_value,
    get_risk_color,
)
from reports import (
    ar_label,
    build_plain_language_report,
    build_technical_report,
    build_technical_json,
)

st.title("Live Solar Flare Predictor")

predictor = load_predictor()
test_data = load_test_data()

active_regions = test_data["NOAA_AR"].unique()

st.sidebar.header("Observation Selection")

selected_ar = st.sidebar.selectbox(
    "Select Active Region (NOAA AR)",
    sorted(active_regions),
    format_func=ar_label,
)

ar_data = (
    test_data[test_data["NOAA_AR"] == selected_ar]
    .sort_values("observation_time")
    .reset_index(drop=True)
)

timestamps = ar_data["observation_time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

selected_time = st.sidebar.selectbox(
    "Select Observation Time",
    timestamps,
    index=len(timestamps) - 1,
)

idx = timestamps.index(selected_time)
observation = ar_data.iloc[[idx]].copy()
historical_data = ar_data.iloc[: idx + 1].copy()

prediction_result = predictor.predict(observation)

flare_prob = float(prediction_result.iloc[0]["flare_probability"])
risk_level = prediction_result.iloc[0]["risk_level"]
actual_target = observation.iloc[0].get("target_24h", None)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Region", f"NOAA {ar_label(selected_ar)}")

with col2:
    st.metric("Observation Time", selected_time)

with col3:
    st.metric("Prediction Horizon", "24 Hours")

st.markdown("---")
st.header("Prediction Status")

c1, c2 = st.columns([1, 2])

with c1:

    st.markdown(f"## {flare_prob * 100:.2f}%\n**Predicted Flare Probability**")

    st.markdown(
        f"""
        ### Risk Level
        <span style="color:{get_risk_color(risk_level)}; font-size:28px; font-weight:bold;">
        {risk_level}
        </span>
        """,
        unsafe_allow_html=True,
    )

    if flare_prob >= 0.01:
        st.success("FLARE RISK DETECTED")
    else:
        st.info("No significant flare risk detected.")

    if actual_target is not None:
        if actual_target == 1:
            st.write("**Observed outcome:** Flare occurred")
        else:
            st.write("**Observed outcome:** No flare")

with c2:

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=flare_prob * 100,
            title={"text": "Predicted Flare Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 1], "color": "green"},
                    {"range": [1, 5], "color": "orange"},
                    {"range": [5, 20], "color": "red"},
                    {"range": [20, 100], "color": "darkred"},
                ],
                "bar": {"color": "black"},
            },
        )
    )

    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))

    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.header("AI Interpretation")

if risk_level == "VERY HIGH RISK":
    st.warning(
        "The model indicates a very high predicted risk of an M/X-class "
        "solar flare within the next 24 hours. This active region "
        "warrants close monitoring."
    )
elif risk_level == "HIGH RISK":
    st.warning("The model indicates an elevated flare risk. Continued monitoring is recommended.")
elif risk_level == "MODERATE RISK":
    st.info("The model indicates a moderate flare risk.")
else:
    st.success("The model indicates a relatively low flare risk.")

st.caption(
    "The displayed probability is the model's predicted probability and "
    "should not be interpreted as a guaranteed physical probability of a flare."
)

st.markdown("---")
st.header("Key Magnetic Indicators")

importance = predictor.model.feature_importances_
features = predictor.features

importance_df = pd.DataFrame({"Feature": features, "Importance": importance})

excluded = {"HARPNUM", "NOAA_AR", "observation_time", "target_24h"}
importance_df = importance_df[~importance_df["Feature"].isin(excluded)]
importance_df = importance_df.sort_values("Importance", ascending=False).head(5)

for _, row in importance_df.iterrows():

    feature = row["Feature"]
    value = observation.iloc[0].get(feature, None)

    if value is None:
        continue

    fc1, fc2, fc3 = st.columns([2, 1, 1])

    with fc1:
        st.write(f"**{readable_feature_name(feature)}**")
    with fc2:
        st.write(f"`{format_value(value)}`")
    with fc3:
        st.write(f"Importance: `{row['Importance']:.4f}`")

st.markdown("---")
st.header(f"Temporal Evolution: AR {ar_label(selected_ar)}")

plot_columns = [c for c in ["TOTUSJH", "TOTPOT", "USFLUX"] if c in historical_data.columns]

hist_plot_data = historical_data[["observation_time"] + plot_columns].copy()

for col in plot_columns:
    std = hist_plot_data[col].std()
    if std != 0:
        hist_plot_data[col] = (hist_plot_data[col] - hist_plot_data[col].mean()) / std

fig_temporal = px.line(
    hist_plot_data, x="observation_time", y=plot_columns,
    title="Normalized Evolution of Magnetic Features",
)

st.plotly_chart(fig_temporal, width="stretch")

# ============================================================
# REPORTS
# ============================================================

st.markdown("---")
st.header("Reports")

st.markdown(
    "Generate a report for the observation selected above. Reports are built "
    "from the live model and evaluation files, and stay tied to this active "
    "region and time -- change the selection and generate again for a new one."
)

report_key = f"{selected_ar}|{selected_time}"
file_stamp = f"AR{ar_label(selected_ar)}_" + selected_time.replace("-", "").replace(":", "").replace(" ", "_")


def generate(state_name, builder, label):
    try:
        with st.spinner(f"Building {label}..."):
            st.session_state[state_name] = {
                "key": report_key,
                "data": builder(observation, historical_data, selected_ar, selected_time),
            }
    except Exception as exc:
        st.session_state.pop(state_name, None)
        st.error(f"Could not build the {label}: {exc}")


rc1, rc2 = st.columns(2)

with rc1:

    st.subheader("Plain-language report")
    st.caption("A short, jargon-free summary for a general reader (PDF).")

    if st.button("Generate plain-language report", key="gen_plain", width="stretch"):
        generate("plain_report", build_plain_language_report, "plain-language report")

    stored = st.session_state.get("plain_report")

    if stored and stored["key"] == report_key:
        st.download_button(
            "Download plain-language report (PDF)",
            data=stored["data"],
            file_name=f"solar_flare_summary_{file_stamp}.pdf",
            mime="application/pdf",
            width="stretch",
        )

with rc2:

    st.subheader("Technical report")
    st.caption("Full model card, evaluation, explanation and limitations for researchers (PDF + JSON).")

    if st.button("Generate technical report", key="gen_tech", width="stretch"):
        generate("tech_report", build_technical_report, "technical report")
        generate("tech_json", build_technical_json, "technical data export")

    stored = st.session_state.get("tech_report")

    if stored and stored["key"] == report_key:
        st.download_button(
            "Download technical report (PDF)",
            data=stored["data"],
            file_name=f"solar_flare_technical_{file_stamp}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    stored_json = st.session_state.get("tech_json")

    if stored_json and stored_json["key"] == report_key:
        st.download_button(
            "Download data (JSON)",
            data=stored_json["data"],
            file_name=f"solar_flare_data_{file_stamp}.json",
            mime="application/json",
            width="stretch",
        )

st.markdown("---")
st.caption(
    "Solar Flare Prediction System | XGBoost + CNN | "
    "SHARP/GOES + Magnetogram Analysis | 24-hour prediction window"
)