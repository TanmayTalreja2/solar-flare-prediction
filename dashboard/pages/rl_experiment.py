import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import PROJECT_ROOT

RL_DIR = PROJECT_ROOT / "results" / "rl_dry_run"
RESULTS = RL_DIR / "rl_results.csv"
CONFIG = RL_DIR / "rl_config.json"
POLICY = RL_DIR / "rl_policy.csv"

st.title("RL Experiment")

st.warning(
    "Exploratory research checkpoint -- not a production forecasting agent and not "
    "used for any score elsewhere in this app."
)

st.markdown(
    "The project paper proposes a reinforcement-learning forecaster with an asymmetric "
    "reward (missing a major flare costs far more than a false alarm). Its only "
    "validation so far was a synthetic toy environment. This experiment is a first "
    "step on real data: a tabular Double Q-learning agent trained on the real 2012 "
    "SHARP features and scored on the same July-December 2012 test period as XGBoost."
)

if not RESULTS.exists():
    st.info("No results yet. Run `python scripts\\rl_dry_run.py` from the project root, then reload this page.")
    st.stop()

results = pd.read_csv(RESULTS)
config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}

if config:

    st.header("Setup")

    st.markdown(
        f"""
- **State:** {len(config.get('state_features', []))} features (the most important XGBoost features from distinct physical parameters), each split into quantile bins from training data: {', '.join(config.get('state_features', []))} -- {config.get('n_states', 'n/a')} discrete states.
- **Actions:** alert or no alert.
- **Reward (TP, TN, FP, FN):** the paper's placeholder weights (5, 1, -1, -8), plus a sweep over a harsher missed-flare penalty. The variant with the best validation TSS is marked as selected.
- **Training:** {config.get('epochs', 'n/a')} passes over the training months, learning rate {config.get('alpha', 'n/a')}, discount {config.get('gamma', 'n/a')}.
- **Data:** {config.get('train_rows', 0):,} training, {config.get('validation_rows', 0):,} validation and {config.get('test_rows', 0):,} test observations, split chronologically.
"""
    )

st.header("Results on the held-out test period")

shown = results.rename(columns={
    "policy": "Policy", "val_tss": "Validation TSS", "test_tss": "Test TSS",
    "test_recall": "Recall", "test_far": "False-alarm rate",
    "test_precision": "Precision", "test_alerts": "Alerts raised",
    "test_roc_auc": "ROC-AUC (score)",
})[["Policy", "Validation TSS", "Test TSS", "Recall", "False-alarm rate", "Precision", "Alerts raised", "ROC-AUC (score)"]]

st.dataframe(
    shown.style.format({
        "Validation TSS": "{:.4f}", "Test TSS": "{:.4f}", "Recall": "{:.4f}",
        "False-alarm rate": "{:.4f}", "Precision": "{:.4f}", "ROC-AUC (score)": "{:.4f}",
    }),
    width="stretch",
    hide_index=True,
)

fig = px.bar(results, x="policy", y="test_tss", color="role", text="test_tss", title="Test TSS by policy")
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(xaxis_title=None, xaxis_tickangle=-20)
st.plotly_chart(fig, width="stretch")

st.caption(
    "TSS (true skill statistic) = recall minus false-alarm rate. 0 means no skill, 1 is perfect. "
    "False-alarm rate here is the share of non-flare observations that triggered an alert."
)

if POLICY.exists():

    policy = pd.read_csv(POLICY)
    alerting = policy[policy["alert"] == 1].sort_values("train_flare_rate", ascending=False)

    st.header("What the selected policy learned")

    st.markdown(
        f"The selected policy raises an alert in {len(alerting)} of {len(policy)} states."
    )

    if len(alerting) > 0:
        st.dataframe(
            alerting[["bins", "n_train", "train_flare_rate", "margin"]].head(15).rename(columns={
                "bins": "State (bin per feature)", "n_train": "Training observations",
                "train_flare_rate": "Training flare rate", "margin": "Q margin (alert - no alert)",
            }),
            width="stretch",
            hide_index=True,
        )

st.header("How to read this honestly")

st.info(
    "The reward depends only on the action and the true outcome, not on how the action "
    "changes later states. The learned policy is therefore equivalent to a cost-sensitive "
    "threshold: alert whenever a state's historical flare rate is high enough that the "
    "reward for alerting beats the reward for staying quiet. The experiment shows the "
    "reward weighting working on real data. It does not show that reinforcement learning "
    "forecasts flares better than XGBoost."
)