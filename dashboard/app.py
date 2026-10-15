import streamlit as st

st.set_page_config(
    page_title="Solar Flare Prediction",
    page_icon="\u2600\ufe0f",
    layout="wide",
)

st.markdown(
    """<style>
.hero{padding:2.6rem 2.2rem;border-radius:18px;margin-bottom:1.4rem;border:1px solid rgba(255,255,255,0.08);background:radial-gradient(circle at 88% 18%,rgba(255,159,28,0.38),rgba(255,159,28,0) 42%),linear-gradient(135deg,#0b0f1a 0%,#121a2b 60%,#1b1233 100%);}
.hero h1{font-size:2.5rem;line-height:1.15;margin:0 0 .7rem 0;padding:0;}
.hero p{font-size:1.08rem;opacity:.85;max-width:44rem;margin:0;}
.card{border:1px solid rgba(255,255,255,0.10);border-radius:14px;padding:1.05rem 1.2rem;background:rgba(255,255,255,0.03);min-height:9.5rem;}
.card h4{margin:0 0 .45rem 0;padding:0;color:#ff9f1c;font-size:1.05rem;}
.card p{margin:0;opacity:.85;font-size:.93rem;line-height:1.45;}
</style>""",
    unsafe_allow_html=True,
)

home_page = st.Page("pages/home.py", title="Home", icon="\U0001F3E0", default=True)
predictor_page = st.Page("pages/predictor.py", title="Predictor", icon="\U0001F6A8")
live_page = st.Page("pages/live_mode.py", title="Live Mode", icon="\U0001F6F0\uFE0F")
insights_page = st.Page("pages/model_insights.py", title="Model Insights", icon="\U0001F4CA")
rl_page = st.Page("pages/rl_experiment.py", title="RL Experiment", icon="\U0001F9EA")
about_page = st.Page("pages/about.py", title="About", icon="\u2139\uFE0F")

nav = st.navigation([home_page, predictor_page, live_page, insights_page, rl_page, about_page])
nav.run()