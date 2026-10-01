import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from common import load_performance_metrics

st.title("Solar Flare Early Warning System")

st.markdown(
    """
    ## Predict Solar Flare Risk Before It Escalates

    This system estimates the probability of an M-class or X-class
    solar flare occurring within the next 24 hours, using real
    magnetic-field observations of the Sun's active regions.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/predictor.py", label="Run Predictor", width="stretch")

with col2:
    st.page_link("pages/model_insights.py", label="Explore the Models", width="stretch")

st.markdown("---")

st.header("How It Works")

st.markdown(
    "Solar flares are sudden releases of magnetic energy. Two independent "
    "models look at different kinds of evidence, and their outputs are "
    "combined into a single risk estimate:"
)

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("### 1. Tabular Model (XGBoost)")
    st.markdown(
        "Reads 17 magnetic-field measurements of an active region "
        "(and how they've changed over the last few hours) and "
        "estimates flare risk from that."
    )

with step2:
    st.markdown("### 2. Image Model (CNN)")
    st.markdown(
        "Looks directly at the magnetogram image of the active region "
        "and learns visual patterns associated with flare activity."
    )

with step3:
    st.markdown("### 3. Fusion")
    st.markdown(
        "Combines both models' probabilities into one final "
        "risk estimate for the active region."
    )

st.markdown("---")

st.header("Current Model Performance")

st.markdown(
    "A quick look -- full details, feature importance, and confusion "
    "matrices are on the Model Insights page."
)

performance_df = load_performance_metrics()

st.dataframe(
    performance_df.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)

st.caption(
    "ROC-AUC and PR-AUC measure how well the model ranks flare risk, "
    "not raw accuracy -- accuracy alone is misleading here since real "
    "flares are rare events."
)
