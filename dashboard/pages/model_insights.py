import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from common import (
    load_predictor,
    load_performance_metrics,
    compute_xgboost_test_confusion,
    CNN_CONFUSION_IMAGE,
    FUSION_PREDICTIONS_PATH,
)

st.title("Model Insights")

st.markdown(
    "This page is for technical review: model architecture, training "
    "features, and evaluation metrics for each component of the system. "
    "Every number here is loaded live from the actual model and "
    "evaluation files -- nothing is hardcoded."
)

st.markdown("---")
st.header("Understanding the Metrics")

st.markdown(
    """
    **ROC-AUC** does not mean accuracy. It measures how well the model
    ranks a random real flare above a random non-flare, across every
    possible decision threshold. 0.5 is a coin flip, 1.0 is perfect.

    **PR-AUC** (Precision-Recall AUC) matters especially here because
    flares are rare (under 5% of observations) -- a model can score
    high accuracy by predicting "no flare" every time, while PR-AUC
    exposes that as a bad model.
    """
)

st.markdown("---")
st.header("Overall Performance")

performance_df = load_performance_metrics()

st.dataframe(
    performance_df.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)

metric = st.radio("Select evaluation metric:", ["ROC-AUC", "PR-AUC"], horizontal=True)

fig_performance = px.bar(
    performance_df, x="Model", y=metric, text=metric, title=f"{metric} Comparison",
)
fig_performance.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig_performance.update_yaxes(range=[0, 1])

st.plotly_chart(fig_performance, width="stretch")

best_row = performance_df.loc[performance_df[metric].idxmax()]
st.success(f"Best {metric}: {best_row['Model']} -- {best_row[metric]:.4f}.")

st.markdown("---")
st.header("XGBoost (Tabular Model)")

predictor = load_predictor()

xgb_metrics = predictor.model_package.get("metrics", {})
best_config = predictor.model_package.get("best_config", {})

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Features Used", len(predictor.features))
with col2:
    st.metric("Training Period", predictor.model_package.get("training_period", "n/a"))
with col3:
    st.metric("Test Period", predictor.model_package.get("testing_period", "n/a"))

if best_config:
    st.caption(
        f"Selected configuration: {best_config.get('name', 'n/a')} "
        f"(max_depth={best_config.get('max_depth', 'n/a')}, "
        f"n_estimators={best_config.get('n_estimators', 'n/a')})"
    )

xgb_confusion = compute_xgboost_test_confusion()

cm1, cm2 = st.columns(2)

with cm1:
    st.markdown("**Confusion Matrix (unseen test period)**")
    cm = xgb_confusion["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Flare", "Actual: Flare"],
        columns=["Predicted: No Flare", "Predicted: Flare"],
    )
    st.dataframe(cm_df, width="stretch")

with cm2:
    st.markdown("**Test-Period Metrics**")
    st.metric("Precision", f"{xgb_confusion['precision']:.4f}")
    st.metric("Recall", f"{xgb_confusion['recall']:.4f}")
    st.metric("F1", f"{xgb_confusion['f1']:.4f}")

st.markdown("**Feature Importance (Top 15)**")

importance = predictor.model.feature_importances_
features = predictor.features

importance_df = pd.DataFrame({"Feature": features, "Importance": importance})

excluded = {"HARPNUM", "NOAA_AR", "observation_time", "target_24h"}
importance_df = importance_df[~importance_df["Feature"].isin(excluded)]
importance_df = importance_df.sort_values("Importance", ascending=False).head(15)

fig_imp = px.bar(
    importance_df, x="Importance", y="Feature", orientation="h",
    title="Top 15 XGBoost Features",
)
fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_imp, width="stretch")

st.markdown("---")
st.header("CNN (Image Model)")

st.markdown(
    "Architecture: ResNet18, modified for single-channel magnetogram "
    "input, trained on cropped magnetogram images of active regions."
)

if CNN_CONFUSION_IMAGE.exists():
    st.image(str(CNN_CONFUSION_IMAGE), caption="CNN Confusion Matrix (test period)")
else:
    st.info("CNN confusion matrix image not found.")

st.markdown("---")
st.header("Fusion Model")

st.markdown(
    "The fusion model is a logistic regression trained on the "
    "validation-period probabilities of both models above, then "
    "evaluated once on the unseen test period."
)

if FUSION_PREDICTIONS_PATH.exists():

    from sklearn.metrics import confusion_matrix

    fusion_df = pd.read_csv(FUSION_PREDICTIONS_PATH)

    y_true = fusion_df["target_24h_label"].astype(int)
    y_pred = fusion_df["fusion_prediction"].astype(int)

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Flare", "Actual: Flare"],
        columns=["Predicted: No Flare", "Predicted: Flare"],
    )

    st.dataframe(cm_df, width="stretch")

else:
    st.info("Fusion prediction file not found.")

st.warning(
    "Known limitation: improving the tabular model's parameter count "
    "raised its standalone ROC-AUC substantially, but this did not "
    "translate into a meaningfully better fusion result -- the CNN "
    "already dominates the current logistic-regression fusion. This "
    "suggests the fusion mechanism itself, not feature availability, is "
    "now the limiting factor. Documented here rather than hidden."
)
