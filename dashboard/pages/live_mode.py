import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import get_risk_color
from live_reports import (
    build_live_json,
    build_live_plain_report,
    build_live_technical_report,
)
from solarflare.realtime.live_pipeline import run_live_predictions_with_context
from solarflare.realtime.prediction_log import log_predictions

GAUGE_MAX_PCT = 30


def pct(value):
    return "n/a" if pd.isna(value) else f"{value * 100:.2f}%"


def hours_ago(ts):
    if pd.isna(ts):
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - ts.to_pydatetime()).total_seconds() / 3600


def risk_gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".3f"},
            title={"text": "XGBoost flare risk"},
            gauge={
                "axis": {"range": [0, GAUGE_MAX_PCT]},
                "steps": [
                    {"range": [0, 1], "color": "green"},
                    {"range": [1, 5], "color": "orange"},
                    {"range": [5, 20], "color": "red"},
                    {"range": [20, GAUGE_MAX_PCT], "color": "darkred"},
                ],
                "bar": {"color": "black"},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def generate(state_key, builder, label, context, ar, time_text, report_key):
    try:
        with st.spinner(f"Building {label}..."):
            st.session_state[state_key] = {
                "key": report_key,
                "data": builder(context, ar, time_text),
            }
    except Exception as exc:
        st.session_state.pop(state_key, None)
        st.error(f"Could not build the {label}: {exc}")


st.title("Live Mode")

st.warning(
    "LIVE DATA -- this page queries current conditions from JSOC, not the "
    "validated 2012 historical test set used elsewhere in this app. The "
    "models were only evaluated on 2012 data, and live scores cannot be "
    "checked until the following 24 hours have passed. Treat them as experimental."
)

st.markdown(
    "Fetches the last 36 hours of near-real-time SHARP data for currently "
    "NOAA-numbered active regions, builds the same features the production "
    "model was trained on, and scores each region. The risk level comes from "
    "the XGBoost tabular model, the same scale used on the Predictor page. "
    "The image model (CNN) and the fusion of both are shown as supporting signals. "
    "Each region can produce a plain-language or technical report."
)

if st.button("Fetch Latest Data", type="primary"):
    with st.spinner("Querying JSOC, downloading magnetograms and scoring regions -- this can take a minute or two..."):
        results, contexts, error = run_live_predictions_with_context()
        st.session_state["live_results"] = results
        st.session_state["live_contexts"] = contexts
        st.session_state["live_error"] = error

        for stale in [k for k in st.session_state if str(k).startswith("live_report_")]:
            del st.session_state[stale]

        if error is None and len(results) > 0:
            log_predictions(results)

results = st.session_state.get("live_results")
contexts = st.session_state.get("live_contexts", {})
error = st.session_state.get("live_error")

if error:
    st.error(error)

elif results is not None and len(results) > 0:

    newest = results["observation_time"].max()
    age = hours_ago(newest)

    st.success(
        f"{len(results)} active regions scored. Newest observation: {newest} UTC"
        + (f" ({age:.1f} hours ago)." if age is not None else ".")
    )

    for _, row in results.iterrows():

        ar = int(row["NOAA_AR"])
        context = contexts.get(ar)
        time_text = str(row["observation_time"])
        report_key = f"{ar}|{time_text}"
        stamp = pd.Timestamp(row["observation_time"]).strftime("%Y%m%d_%H%M%S")

        with st.container(border=True):

            left, right = st.columns([3, 2])

            with left:
                st.markdown(f"### NOAA {ar} (HARP {row['HARPNUM']})")
                st.caption(f"Observation: {row['observation_time']} UTC | {row['data_series']}")

                st.markdown(
                    f"<span style='color:{get_risk_color(row['risk_level'])}; "
                    f"font-size:24px; font-weight:bold;'>{row['risk_level']}</span>",
                    unsafe_allow_html=True,
                )

                st.write(f"XGBoost (risk basis): {pct(row['xgb_probability'])}")
                st.write(f"CNN (image): {pct(row['cnn_probability'])}")
                st.write(f"Fusion: {pct(row['fusion_probability'])}")

                if not pd.isna(row["image_time"]):
                    st.caption(f"Magnetogram time: {row['image_time']} UTC")

                if row["note"]:
                    st.caption(row["note"])

            with right:
                st.plotly_chart(risk_gauge(row["xgb_probability"]), width="stretch", key=f"gauge_{ar}")

            if context is None:
                st.caption("Report data for this region is not available.")
                continue

            plain_key = f"live_report_plain_{ar}"
            tech_key = f"live_report_tech_{ar}"
            json_key = f"live_report_json_{ar}"

            r1, r2 = st.columns(2)

            with r1:

                if st.button("Generate plain-language report", key=f"gen_plain_{ar}", width="stretch"):
                    generate(plain_key, build_live_plain_report, "plain-language report",
                             context, ar, time_text, report_key)

                stored = st.session_state.get(plain_key)

                if stored and stored["key"] == report_key:
                    st.download_button(
                        "Download plain-language report (PDF)",
                        data=stored["data"],
                        file_name=f"live_summary_AR{ar}_{stamp}.pdf",
                        mime="application/pdf",
                        key=f"dl_plain_{ar}",
                        width="stretch",
                    )

            with r2:

                if st.button("Generate technical report", key=f"gen_tech_{ar}", width="stretch"):
                    generate(tech_key, build_live_technical_report, "technical report",
                             context, ar, time_text, report_key)
                    generate(json_key, build_live_json, "technical data export",
                             context, ar, time_text, report_key)

                stored = st.session_state.get(tech_key)

                if stored and stored["key"] == report_key:
                    st.download_button(
                        "Download technical report (PDF)",
                        data=stored["data"],
                        file_name=f"live_technical_AR{ar}_{stamp}.pdf",
                        mime="application/pdf",
                        key=f"dl_tech_{ar}",
                        width="stretch",
                    )

                stored_json = st.session_state.get(json_key)

                if stored_json and stored_json["key"] == report_key:
                    st.download_button(
                        "Download data (JSON)",
                        data=stored_json["data"],
                        file_name=f"live_data_AR{ar}_{stamp}.json",
                        mime="application/json",
                        key=f"dl_json_{ar}",
                        width="stretch",
                    )

    st.caption(
        f"Gauges span 0-{GAUGE_MAX_PCT}% so small live scores stay visible. The CNN and fusion models were "
        "trained on a class-balanced image sample (half flare, half no-flare), so their percentages read "
        "much higher than XGBoost's and are not comparable to the risk bands."
    )

elif results is not None:
    st.info("No NOAA-numbered active regions found in the live data right now.")

else:
    st.info("Click \"Fetch Latest Data\" to run a live prediction.")