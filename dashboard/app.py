import streamlit as st

st.set_page_config(
    page_title="Solar Flare Prediction",
    page_icon="\u2600\ufe0f",
    layout="wide",
)

home_page = st.Page("pages/home.py", title="Home", icon="\U0001F3E0", default=True)
predictor_page = st.Page("pages/predictor.py", title="Predictor", icon="\U0001F6A8")
insights_page = st.Page("pages/model_insights.py", title="Model Insights", icon="\U0001F4CA")

nav = st.navigation([home_page, predictor_page, insights_page])
nav.run()
