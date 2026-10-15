import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from common import load_performance_metrics, load_predictor, load_test_data


def card(title, body):
    return f'<div class="card"><h4>{title}</h4><p>{body}</p></div>'


st.markdown(
    '<div class="hero"><h1>Predict Solar Flare Risk Before It Escalates</h1>'
    "<p>Machine-learning models read the magnetic state of the Sun's active regions "
    "and estimate the risk of a strong (M- or X-class) solar flare within the next 24 hours.</p></div>",
    unsafe_allow_html=True,
)

b1, b2, b3 = st.columns(3)

with b1:
    st.page_link("pages/predictor.py", label="Run Predictor", width="stretch")
with b2:
    st.page_link("pages/live_mode.py", label="Try Live Mode", width="stretch")
with b3:
    st.page_link("pages/model_insights.py", label="Explore the Models", width="stretch")

predictor = load_predictor()
test_data = load_test_data()

s1, s2, s3, s4 = st.columns(4)
s1.metric("Model features", len(predictor.features))
s2.metric("Test observations", f"{len(test_data):,}")
s3.metric("Active regions in test", int(test_data["NOAA_AR"].nunique()))
s4.metric("Prediction window", "24 hours")

st.header("How it works")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(card(
        "1. Tabular model (XGBoost)",
        "Reads 17 magnetic-field measurements of an active region and how they changed over the last few hours.",
    ), unsafe_allow_html=True)

with c2:
    st.markdown(card(
        "2. Image model (CNN)",
        "Looks at the region's magnetogram, an image of the magnetic field, for visual patterns linked to flares.",
    ), unsafe_allow_html=True)

with c3:
    st.markdown(card(
        "3. Fusion",
        "Combines both models' outputs into one estimate for regions where both an image and measurements exist.",
    ), unsafe_allow_html=True)

st.header("Why it matters")

w1, w2, w3, w4 = st.columns(4)

with w1:
    st.markdown(card("Navigation", "Strong flares can briefly reduce GPS accuracy."), unsafe_allow_html=True)
with w2:
    st.markdown(card("Radio", "They can black out high-frequency radio used by aircraft, ships and emergency services."), unsafe_allow_html=True)
with w3:
    st.markdown(card("Satellites", "Radiation and charged particles can upset satellites, which operators may protect by switching to safe modes."), unsafe_allow_html=True)
with w4:
    st.markdown(card("Power and flights", "In extreme cases, grid operators and airlines take precautions."), unsafe_allow_html=True)

st.header("Current model performance")

performance_df = load_performance_metrics()

st.dataframe(
    performance_df.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)

st.caption(
    "ROC-AUC and PR-AUC measure how well a model ranks flare risk, not raw accuracy. "
    "Accuracy alone would be misleading because real flares are rare."
)

st.info(
    "This is a research prototype trained and tested on 2012 data. It ranks risk better than "
    "chance, but it misses many flares and raises many false alarms. Treat it as one signal "
    "among many, never as a forecast to rely on by itself."
)