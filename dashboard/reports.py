"""PDF / JSON report generation for the Solar Flare dashboard.

Every number in the reports comes from the loaded model, the results
files, or the selected observation. Nothing is hardcoded except
static methodology notes and limitations, which are clearly worded.
"""

import io
import json
import sys
from datetime import datetime
from importlib import metadata
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from common import (
    CNN_METRICS_PATH,
    FEATURES_PATH,
    FUSION_METRICS_PATH,
    FUSION_PREDICTIONS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    compute_xgboost_test_confusion,
    format_value,
    load_data,
    load_performance_metrics,
    load_predictor,
    load_test_data,
)

HYPERPARAM_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "xgboost_25params" / "hyperparameter_results.csv"
)

# These bands mirror SolarFlarePredictor.classify_risk in scripts/predict.py
RISK_BANDS = [
    ("LOW RISK", 0.0, 0.01, "#2e7d32"),
    ("MODERATE RISK", 0.01, 0.05, "#ef6c00"),
    ("HIGH RISK", 0.05, 0.20, "#c62828"),
    ("VERY HIGH RISK", 0.20, 1.0, "#7f0000"),
]
RISK_HEX = {label: color for label, _, _, color in RISK_BANDS}

FOOTER_TEXT = (
    "Solar Flare Prediction System - research prototype - "
    "not an operational forecast"
)


# ============================================================
# FEATURE DESCRIPTIONS (standard SHARP parameter meanings)
# ============================================================

BASE_INFO = {
    "USFLUX": ("Total unsigned magnetic flux", "how much magnetic field passes through the active region overall"),
    "MEANGAM": ("Mean field inclination angle", "how tilted the magnetic field is relative to straight up from the Sun's surface"),
    "MEANGBT": ("Mean gradient of total field", "how sharply the magnetic field strength changes from place to place"),
    "MEANGBZ": ("Mean gradient of vertical field", "how sharply the upward-pointing part of the field changes across the region"),
    "MEANGBH": ("Mean gradient of horizontal field", "how sharply the sideways part of the field changes across the region"),
    "MEANJZD": ("Mean vertical current density", "how strongly electric current flows up through the region on average"),
    "TOTUSJZ": ("Total unsigned vertical current", "the total amount of electric current flowing through the region"),
    "MEANALP": ("Mean twist parameter", "how twisted the magnetic field lines are on average"),
    "MEANJZH": ("Mean current helicity", "a measure of how twisted the region's electric currents are, on average"),
    "TOTUSJH": ("Total unsigned current helicity", "the overall amount of twist in the region's magnetic field"),
    "ABSNJZH": ("Absolute net current helicity", "the net twist of the field, ignoring its direction"),
    "SAVNCPP": ("Sum of net current per polarity", "how much net electric current is carried by each magnetic polarity"),
    "MEANPOT": ("Mean free magnetic energy", "the average stored magnetic energy available to be released"),
    "TOTPOT": ("Total free magnetic energy", "the total stored magnetic energy that could power a flare"),
    "MEANSHR": ("Mean shear angle", "how strongly the field is sheared (skewed) along the boundary between opposite polarities"),
    "SHRGT45": ("Highly sheared area fraction", "the fraction of the region where the shear angle exceeds 45 degrees"),
    "R_VALUE": ("Polarity inversion line flux (R value)", "how much magnetic flux sits near the boundary between opposite polarities, a known flare indicator"),
    "observation_hour": ("Hour of day", "the hour of the day of the observation"),
    "day_of_year": ("Day of year", "the position of the observation in the calendar year"),
}

WINDOW_TEXT = {"1h": "1 hour", "3h": "3 hours", "6h": "6 hours", "12h": "12 hours"}


def describe_feature(name):
    """Return {'title': ..., 'plain': ...} for any model feature name."""

    base = name
    log_scale = False
    suffix = ""

    if base.startswith("LOG_"):
        log_scale = True
        base = base[4:]

    markers = (
        ("_RELCHANGE_", "relative change over the past {w}"),
        ("_CHANGE_", "change over the past {w}"),
        ("_ROLLSTD_", "variability over the past {w}"),
    )

    for marker, template in markers:
        if marker in base:
            base, window = base.split(marker, 1)
            suffix = template.format(w=WINDOW_TEXT.get(window, window))
            break

    info = BASE_INFO.get(base)

    if info is None:
        title = base.replace("_", " ").title()
        meaning = "a measurement used by the model"
    else:
        title, meaning = info

    if log_scale:
        title = f"{title} (log scale)"

    if suffix:
        return {"title": f"{title}: {suffix}", "plain": f"{meaning}; here, the {suffix}"}

    return {"title": title, "plain": meaning}


# ============================================================
# SMALL HELPERS
# ============================================================

def ar_label(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _fmt(value, digits=4):
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _score_text(prob):
    if prob < 0.0001:
        return "<0.01%"
    return f"{prob * 100:.2f}%"


def _ratio_text(fraction):
    """Turn a small fraction into 'roughly 1 in N' phrasing."""

    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        return "not enough data to tell"

    if fraction <= 0:
        return "essentially never, in this test period"

    if fraction >= 1:
        return "essentially every time, in this test period"

    n = round(1 / fraction)

    return f"roughly 1 in {n}"


def _perf_row(perf, prefix):
    match = perf[perf["Model"].str.startswith(prefix)]
    return None if match.empty else match.iloc[0]


def _feature_groups(features):
    change = [f for f in features if "_RELCHANGE_" not in f and "_CHANGE_" in f]
    rel = [f for f in features if "_RELCHANGE_" in f]
    roll = [f for f in features if "_ROLLSTD_" in f]
    logf = [f for f in features if f.startswith("LOG_")]
    cal = [f for f in features if f in ("observation_hour", "day_of_year")]
    used = set(change + rel + roll + logf + cal)
    base = [f for f in features if f not in used]
    return {
        "base": base,
        "log": logf,
        "calendar": cal,
        "change": change,
        "relchange": rel,
        "rollstd": roll,
    }


def _version(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return "unknown"


# ============================================================
# PER-OBSERVATION EXPLANATION (XGBoost TreeSHAP contributions)
# ============================================================

def explain_observation(predictor, observation):
    """Return (contribution_df, baseline_logodds, n_imputed).

    Uses XGBoost's native pred_contribs output (TreeSHAP), in log-odds
    units. Returns (None, None, n_imputed) if it cannot be computed.
    """

    import xgboost as xgb

    raw = observation[predictor.features].astype(float)
    n_imputed = int(raw.isna().sum().sum())

    try:
        x_imp = predictor.imputer.transform(raw)
        contribs = predictor.model.get_booster().predict(
            xgb.DMatrix(x_imp), pred_contribs=True
        )
        contribs = np.asarray(contribs)[0]
    except Exception:
        return None, None, n_imputed

    frame = pd.DataFrame({
        "feature": predictor.features,
        "contribution": contribs[:-1],
        "value": raw.iloc[0].values,
        "imputed": raw.iloc[0].isna().values,
    })

    frame["abs"] = frame["contribution"].abs()
    frame = frame.sort_values("abs", ascending=False).reset_index(drop=True)

    return frame, float(contribs[-1]), n_imputed


# ============================================================
# CONTEXT COLLECTION
# ============================================================

def _fusion_details():

    from sklearn.metrics import confusion_matrix

    out = {}

    if FUSION_METRICS_PATH.exists():
        df = pd.read_csv(FUSION_METRICS_PATH)
        row = df[df["model"] == "Fusion"]
        if not row.empty:
            out["metrics"] = row.iloc[0].to_dict()

    if FUSION_PREDICTIONS_PATH.exists():
        preds = pd.read_csv(FUSION_PREDICTIONS_PATH)
        y_true = preds["target_24h_label"].astype(int)
        y_pred = preds["fusion_prediction"].astype(int)
        out["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
        out["n"] = int(len(preds))

    return out


def _cnn_details():

    if not CNN_METRICS_PATH.exists():
        return None

    return pd.read_csv(CNN_METRICS_PATH).iloc[0].to_dict()


def _collect(observation, history, selected_ar, selected_time):

    predictor = load_predictor()
    test_data = load_test_data()

    result = predictor.predict(observation).iloc[0]
    row = observation.iloc[0]

    target = row.get("target_24h", None)
    actual = None if target is None or pd.isna(target) else int(target)

    contrib_df, baseline, n_imputed = explain_observation(predictor, observation)

    harp = row.get("HARPNUM", None)

    return {
        "predictor": predictor,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ar": ar_label(selected_ar),
        "harp": None if harp is None or pd.isna(harp) else ar_label(harp),
        "time": str(selected_time),
        "prob": float(result["flare_probability"]),
        "risk": str(result["risk_level"]),
        "pred_class": int(result["prediction"]),
        "threshold": float(predictor.threshold),
        "actual": actual,
        "base_rate": float(test_data["target_24h"].mean()),
        "n_test": int(len(test_data)),
        "perf": load_performance_metrics(),
        "xgb_conf": compute_xgboost_test_confusion(),
        "observation": observation,
        "history": history,
        "contrib": contrib_df,
        "baseline": baseline,
        "n_imputed": n_imputed,
        "fusion": _fusion_details(),
        "cnn": _cnn_details(),
    }


# ============================================================
# CHARTS (matplotlib Figure objects, no pyplot state)
# ============================================================

def _img(fig, width_cm):

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)

    iw, ih = ImageReader(buf).getSize()
    buf.seek(0)

    width = width_cm * cm

    return Image(buf, width=width, height=width * ih / iw)


def chart_risk_ladder(prob):

    fig = Figure(figsize=(7, 1.7))
    ax = fig.subplots()

    edges = ["under 1%", "1% - 5%", "5% - 20%", "20% and above"]

    for i, (label, lo, hi, color) in enumerate(RISK_BANDS):
        ax.barh(0, 1, left=i, color=color, edgecolor="white", height=0.5)
        ax.text(i + 0.5, 0, label.replace(" RISK", ""), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(i + 0.5, -0.42, edges[i], ha="center", va="top", fontsize=7.5, color="#444444")

    pos = 0.0

    for i, (label, lo, hi, color) in enumerate(RISK_BANDS):
        if lo <= prob < hi or i == len(RISK_BANDS) - 1:
            pos = i + (min(prob, hi) - lo) / (hi - lo)
            break

    ax.annotate("", xy=(pos, 0.27), xytext=(pos, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(pos, 0.78, _score_text(prob), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(-0.75, 1.25)
    ax.axis("off")

    return fig


def chart_contributions(frame, n=10):

    top = frame.head(n).iloc[::-1]

    fig = Figure(figsize=(7, 0.42 * len(top) + 0.9))
    ax = fig.subplots()

    bar_colors = ["#c62828" if v > 0 else "#1565c0" for v in top["contribution"]]

    labels = []
    for f in top["feature"]:
        title = describe_feature(f)["title"]
        labels.append(title if len(title) <= 60 else title[:57] + "...")

    ax.barh(labels, top["contribution"], color=bar_colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution to the model score (log-odds). Red raises risk, blue lowers it.", fontsize=8)
    ax.tick_params(labelsize=7.5)

    return fig


def chart_temporal(history):

    cols = [c for c in ("TOTUSJH", "TOTPOT", "USFLUX", "R_VALUE") if c in history.columns]

    if not cols or len(history) < 3:
        return None

    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()

    for c in cols:
        series = history[c].astype(float)
        std = series.std()
        if std and not np.isnan(std):
            z = (series - series.mean()) / std
        else:
            z = series * 0
        ax.plot(history["observation_time"], z, label=BASE_INFO.get(c, (c, ""))[0], lw=1.3)

    ax.legend(fontsize=7, loc="best")
    ax.set_ylabel("Standardised value", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.autofmt_xdate()

    return fig


def chart_confusions(items):

    n = len(items)

    fig = Figure(figsize=(3.4 * n, 3.2))
    axes = fig.subplots(1, n, squeeze=False)[0]

    for ax, (title, cm_) in zip(axes, items):

        ax.imshow(cm_, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No flare", "Flare"], fontsize=8)
        ax.set_yticklabels(["No flare", "Flare"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(title, fontsize=9)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm_[i, j]):,}", ha="center", va="center", fontsize=9,
                        color="white" if cm_[i, j] > cm_.max() / 2 else "black")

    fig.tight_layout()

    return fig


def chart_metric_bars(perf):

    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()

    x = np.arange(len(perf))
    width = 0.35

    roc = perf["ROC-AUC"].astype(float).values
    pr = perf["PR-AUC"].astype(float).values

    b1 = ax.bar(x - width / 2, roc, width, label="ROC-AUC", color="#1565c0")
    b2 = ax.bar(x + width / 2, pr, width, label="PR-AUC", color="#ef6c00")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(perf["Model"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    return fig


def chart_global_importance(predictor, n=15):

    frame = pd.DataFrame({
        "feature": predictor.features,
        "importance": predictor.model.feature_importances_,
    }).sort_values("importance", ascending=False).head(n).iloc[::-1]

    fig = Figure(figsize=(7, 0.34 * len(frame) + 0.9))
    ax = fig.subplots()

    labels = []
    for f in frame["feature"]:
        title = describe_feature(f)["title"]
        labels.append(title if len(title) <= 60 else title[:57] + "...")

    ax.barh(labels, frame["importance"], color="#1565c0")
    ax.set_xlabel("XGBoost feature importance", fontsize=8)
    ax.tick_params(labelsize=7.5)

    return fig


# ============================================================
# PDF BUILDING BLOCKS
# ============================================================

def _styles():

    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle("RTitle", parent=base["Title"], fontSize=22, leading=26,
                                alignment=0, spaceAfter=4),
        "h1": ParagraphStyle("RH1", parent=base["Heading1"], fontSize=15, spaceBefore=14,
                             spaceAfter=6, textColor=colors.HexColor("#1a237e")),
        "h2": ParagraphStyle("RH2", parent=base["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("RBody", parent=base["BodyText"], fontSize=10, leading=14, spaceAfter=6),
        "small": ParagraphStyle("RSmall", parent=base["BodyText"], fontSize=8.5, leading=11,
                                textColor=colors.HexColor("#555555")),
        "cell": ParagraphStyle("RCell", parent=base["BodyText"], fontSize=8.5, leading=10.5),
        "cellb": ParagraphStyle("RCellB", parent=base["BodyText"], fontSize=8.5, leading=10.5,
                                fontName="Helvetica-Bold"),
        "bullet": ParagraphStyle("RBullet", parent=base["BodyText"], fontSize=10, leading=14,
                                 leftIndent=14, bulletIndent=2, spaceAfter=3),
        "risk_label": ParagraphStyle("RRiskLabel", parent=base["BodyText"], fontSize=10, leading=14,
                                     alignment=1, textColor=colors.white),
        "risk_value": ParagraphStyle("RRiskValue", parent=base["BodyText"], fontSize=30, leading=36,
                                     alignment=1, textColor=colors.white, fontName="Helvetica-Bold"),
        "risk_name": ParagraphStyle("RRiskName", parent=base["BodyText"], fontSize=15, leading=20,
                                    alignment=1, textColor=colors.white, fontName="Helvetica-Bold"),
    }


def _table(rows, col_widths, S, header=True):

    data = []

    for r_i, row in enumerate(rows):
        style = S["cellb"] if (header and r_i == 0) else S["cell"]
        data.append([Paragraph(escape(str(cell)), style) for cell in row])

    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)

    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")))

    table.setStyle(TableStyle(commands))

    return table


def _risk_box(ctx, S):

    table = Table(
        [
            [Paragraph("MODEL RISK SCORE", S["risk_label"])],
            [Paragraph(escape(_score_text(ctx["prob"])), S["risk_value"])],
            [Paragraph(escape(ctx["risk"]), S["risk_name"])],
        ],
        colWidths=[9 * cm],
        hAlign="CENTER",
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(RISK_HEX.get(ctx["risk"], "#555555"))),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return table


def _render(story, title):

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title=title,
        author="Solar Flare Prediction System",
    )

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.1 * cm, FOOTER_TEXT)
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    return buf.getvalue()


# ============================================================
# PLAIN-LANGUAGE REPORT CONTENT
# ============================================================

WHAT_THE_MODEL_SAW = {
    "LOW RISK": "The model does not see much of the pattern it has learned to associate with strong flares in this region right now.",
    "MODERATE RISK": "The model sees a little of the pattern it has learned to associate with strong flares in this region.",
    "HIGH RISK": "The model sees a fair amount of the pattern it has learned to associate with strong flares in this region.",
    "VERY HIGH RISK": "The model sees a strong version of the pattern it has learned to associate with strong flares in this region.",
}

DAILY_IMPACT = {
    "LOW RISK": (
        "At this level, you would almost certainly not notice anything in daily life. GPS, phone "
        "networks, radio, and satellite services are expected to work as normal."
    ),
    "MODERATE RISK": (
        "At this level, most people would not notice anything. A few sensitive systems might see "
        "very minor effects, such as brief static on long-distance radio, but nothing most people "
        "would ever notice."
    ),
    "HIGH RISK": (
        "At this level, some real-world effects become more likely, though most people still would "
        "not notice. Long-distance radio (used by aircraft, ships, and emergency services) can lose "
        "signal for a while on the sunlit side of Earth, GPS positioning can become slightly less "
        "accurate, and airlines occasionally reroute flights that cross near the poles as a precaution."
    ),
    "VERY HIGH RISK": (
        "At this level, effects are more likely to be noticeable to people outside specialist fields. "
        "Radio blackouts affecting aircraft, ships, and emergency communication can last longer and "
        "cover a wider area, GPS errors can be more significant, satellite operators may switch "
        "spacecraft into a protective mode, astronauts and high-altitude flight crews can receive "
        "extra radiation exposure, and power-grid operators in some regions may take precautionary "
        "steps against voltage swings."
    ),
}

GLOSSARY = [
    ("Active region", "A patch of the Sun's surface with a strong, complex magnetic field, where flares originate."),
    ("Magnetogram", "An image showing the strength and direction of the Sun's magnetic field across an active region."),
    ("Magnetic flux", "How much magnetic field is present in a region -- more flux generally means a bigger, more complex region."),
    ("Magnetic energy", "Energy stored in a region's twisted, stressed magnetic field; a flare releases some of this energy suddenly."),
    ("Electric current", "A flow of charged particles through the Sun's atmosphere; strong currents often accompany flare-prone regions."),
    ("Shear / twist", "How skewed or twisted the magnetic field lines are compared to a simple, relaxed shape; more of either usually means more stored energy."),
    ("Polarity inversion line", "The boundary between areas of opposite magnetic polarity; flares often start near this boundary."),
    ("M-class / X-class flare", "The two strongest categories on the standard flare scale; X-class is the most powerful and has the largest real-world effects."),
    ("Risk score / risk band", "The model's output for this region: a relative warning level, not a literal percentage chance of a flare."),
    ("Radio blackout", "A temporary loss of high-frequency radio signal, mainly affecting aviation, shipping, and emergency communication on the sunlit side of Earth."),
    ("GPS / satellite navigation error", "A temporary drop in GPS positioning accuracy, caused by disturbances the flare creates in Earth's upper atmosphere."),
    ("Satellite anomaly", "An unexpected malfunction or protective shutdown of a satellite, sometimes triggered by radiation or charged particles from solar activity."),
]


# ============================================================
# PLAIN-LANGUAGE REPORT
# ============================================================

def build_plain_language_report(observation, history, selected_ar, selected_time):

    ctx = _collect(observation, history, selected_ar, selected_time)
    S = _styles()
    story = []

    story.append(Paragraph("Solar Flare Risk Summary", S["title"]))
    story.append(Paragraph(
        escape(f"Active region NOAA {ctx['ar']}  |  Observation time {ctx['time']}  |  "
               f"Report generated {ctx['generated']}"),
        S["small"],
    ))
    story.append(Spacer(1, 10))

    story.append(_risk_box(ctx, S))
    story.append(Spacer(1, 6))
    story.append(_img(chart_risk_ladder(ctx["prob"]), 15))

    story.append(Paragraph("The short version", S["h1"]))
    story.append(Paragraph(
        escape(
            f"For this active region at this time, the model's risk score is "
            f"{_score_text(ctx['prob'])}, placing it in the {ctx['risk'].lower()} band. "
            f"{WHAT_THE_MODEL_SAW.get(ctx['risk'], '')} Words in italics below are explained "
            f"in the glossary at the end of this report."
        ),
        S["body"],
    ))

    if ctx["actual"] is not None:
        outcome = "did" if ctx["actual"] == 1 else "did not"
        story.append(Paragraph(
            escape(
                f"This is a historical record from 2012, so we can check what actually happened "
                f"next: a strong flare {outcome} occur in this region within the following 24 "
                f"hours. In real-time use, this information would not yet be known."
            ),
            S["body"],
        ))

    story.append(Paragraph("What could this mean for day-to-day life?", S["h1"]))
    story.append(Paragraph(escape(DAILY_IMPACT.get(ctx["risk"], "")), S["body"]))

    story.append(Paragraph("What led to this reading", S["h1"]))
    story.append(Paragraph(
        escape(
            "The model looks at measurements of this region's magnetic field, and how those "
            "measurements have changed over the last few hours. These mattered most for this "
            "particular reading:"
        ),
        S["body"],
    ))

    if ctx["contrib"] is not None:

        for _, row in ctx["contrib"].head(5).iterrows():

            info = describe_feature(row["feature"])
            direction = "increased the risk score" if row["contribution"] > 0 else "decreased the risk score"
            note = " (this measurement was missing here and was filled in with a typical value)" if row["imputed"] else ""

            text = f"<b>{escape(info['title'])}</b> -- this {direction}. This measures {escape(info['plain'])}.{escape(note)}"
            story.append(Paragraph(text, S["bullet"], bulletText="\u2022"))

    else:

        importance = pd.DataFrame({
            "feature": ctx["predictor"].features,
            "importance": ctx["predictor"].model.feature_importances_,
        }).sort_values("importance", ascending=False).head(5)

        story.append(Paragraph(
            escape("A breakdown for this specific reading was not available, so these are the "
                   "measurements the model relies on most in general:"),
            S["body"],
        ))

        for f in importance["feature"]:
            info = describe_feature(f)
            text = f"<b>{escape(info['title'])}</b> -- this measures {escape(info['plain'])}."
            story.append(Paragraph(text, S["bullet"], bulletText="\u2022"))

    temporal = chart_temporal(ctx["history"])

    if temporal is not None:
        story.append(Spacer(1, 6))
        story.append(KeepTogether([
            _img(temporal, 15),
            Paragraph(escape("How four key measurements of this region changed in the period "
                             "leading up to the reading (each rescaled to be comparable)."),
                      S["small"]),
        ]))

    story.append(Paragraph("How much should you trust this?", S["h1"]))

    precision = ctx["xgb_conf"]["precision"]
    recall = ctx["xgb_conf"]["recall"]

    story.append(Paragraph(
        escape(
            f"This tool is a research prototype, tested on one year of historical data (2012). "
            f"When it raised an alert like this one, {_ratio_text(precision)} of those alerts "
            f"were actually followed by a real flare. And out of all the real flares that happened "
            f"in the test year, the model only caught {_ratio_text(recall)} of them -- it "
            f"missed most of them."
        ),
        S["body"],
    ))
    story.append(Paragraph(
        escape(
            "In short: treat this as one extra signal, not a forecast you can rely on by itself. "
            "Most alerts will not lead to a flare, and many flares will happen without an alert."
        ),
        S["body"],
    ))
    story.append(Paragraph(
        escape(
            "Note: this score comes from the model that reads magnetic-field measurements. A "
            "separate model that looks at solar images exists in the project but is not used for "
            "this particular score."
        ),
        S["small"],
    ))

    story.append(Paragraph("What is a solar flare?", S["h1"]))
    story.append(Paragraph(
        escape(
            "A solar flare is a sudden release of magnetic energy from the Sun's surface. The "
            "strongest ones (M-class and X-class) can disturb radio communication and GPS "
            "accuracy, affect satellites, and in extreme cases stress power grids or raise "
            "radiation exposure for high-altitude flights. Forecasting them is difficult, which is "
            "why this tool reports a risk level rather than a certainty."
        ),
        S["body"],
    ))

    story.append(PageBreak())
    story.append(Paragraph("Glossary", S["h1"]))
    story.append(Paragraph(
        escape("Plain-language meanings for the technical terms used in this report."),
        S["small"],
    ))
    story.append(Spacer(1, 4))

    glossary_rows = [["Term", "Meaning"]] + [list(pair) for pair in GLOSSARY]
    story.append(_table(glossary_rows, [4.5 * cm, 12.5 * cm], S))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        escape("This report is produced by a research prototype and must not be used as the sole "
               "basis for any operational decision."),
        S["small"],
    ))

    return _render(story, "Solar Flare Risk Summary")


# ============================================================
# TECHNICAL REPORT
# ============================================================

def build_technical_report(observation, history, selected_ar, selected_time):

    ctx = _collect(observation, history, selected_ar, selected_time)
    predictor = ctx["predictor"]
    package = predictor.model_package
    S = _styles()
    story = []

    full_w = 17 * cm

    story.append(Paragraph("Solar Flare Prediction: Technical Report", S["title"]))
    story.append(Paragraph(
        escape(f"Active region NOAA {ctx['ar']}  |  Observation {ctx['time']}  |  "
               f"Generated {ctx['generated']}"),
        S["small"],
    ))

    # ---------------- 1. Prediction summary ----------------
    story.append(Paragraph("1. Prediction summary", S["h1"]))

    rows = [
        ["Item", "Value"],
        ["NOAA active region", ctx["ar"]],
        ["HARP number", ctx["harp"] or "n/a"],
        ["Observation time (TAI)", ctx["time"]],
        ["Model score (flare probability output)", _score_text(ctx["prob"])],
        ["Risk band", ctx["risk"]],
        ["Decision threshold (saved with model)", _fmt(ctx["threshold"], 3)],
        ["Predicted class at threshold", "Flare (1)" if ctx["pred_class"] == 1 else "No flare (0)"],
        ["Observed label (retrospective, unavailable in live use)",
         "n/a" if ctx["actual"] is None else ("Flare within 24h (1)" if ctx["actual"] == 1 else "No flare within 24h (0)")],
        ["Features imputed for this observation", f"{ctx['n_imputed']} of {len(predictor.features)}"],
        ["Test-period base rate of positives", f"{ctx['base_rate'] * 100:.2f}%"],
    ]
    story.append(_table(rows, [7.5 * cm, 9.5 * cm], S))
    story.append(Spacer(1, 6))
    story.append(_img(chart_risk_ladder(ctx["prob"]), 14))

    # ---------------- 2. Per-observation explanation ----------------
    story.append(Paragraph("2. Per-observation explanation", S["h1"]))

    if ctx["contrib"] is not None:

        story.append(Paragraph(
            escape(
                "Contributions are XGBoost's native TreeSHAP values in log-odds units for this "
                "observation. Positive values raise the model score, negative values lower it. "
                "The baseline is the model's average log-odds; baseline plus all contributions "
                "gives this observation's final log-odds."
            ),
            S["body"],
        ))

        total = float(ctx["contrib"]["contribution"].sum())

        story.append(_table([
            ["Baseline log-odds", "Sum of all contributions", "Final log-odds"],
            [_fmt(ctx["baseline"], 3), _fmt(total, 3), _fmt(ctx["baseline"] + total, 3)],
        ], [full_w / 3] * 3, S))
        story.append(Spacer(1, 6))

        story.append(_img(chart_contributions(ctx["contrib"], 10), 15))
        story.append(Spacer(1, 6))

        top = ctx["contrib"].head(10)
        rows = [["#", "Feature", "Meaning", "Value", "Contribution"]]

        for i, (_, r) in enumerate(top.iterrows(), start=1):
            info = describe_feature(r["feature"])
            value = format_value(r["value"]) + (" (imputed)" if r["imputed"] else "")
            rows.append([i, r["feature"], info["plain"], value, _fmt(r["contribution"], 3)])

        story.append(_table(rows, [0.8 * cm, 4.4 * cm, 6.4 * cm, 2.9 * cm, 2.5 * cm], S))

    else:
        story.append(Paragraph(
            escape("Per-observation contributions could not be computed for this observation."),
            S["body"],
        ))

    temporal = chart_temporal(ctx["history"])

    if temporal is not None:
        story.append(Spacer(1, 8))
        story.append(KeepTogether([
            _img(temporal, 15),
            Paragraph(escape("Standardised evolution of selected parameters for this active region "
                             "up to the observation time."), S["small"]),
        ]))

    story.append(PageBreak())

    # ---------------- 3. Model card ----------------
    story.append(Paragraph("3. Model card: XGBoost tabular classifier", S["h1"]))

    groups = _feature_groups(predictor.features)
    config = package.get("best_config", {}) or {}

    rows = [
        ["Item", "Value"],
        ["Model type", "XGBClassifier (gradient-boosted decision trees), objective binary:logistic"],
        ["Prediction target", "M- or X-class flare starting within 24 hours of the observation, per active region"],
        ["Total features", str(len(predictor.features))],
        ["  Base SHARP parameters", f"{len(groups['base'])}: " + ", ".join(groups["base"])],
        ["  Log-scaled features", f"{len(groups['log'])}: " + ", ".join(groups["log"])],
        ["  Calendar features", ", ".join(groups["calendar"]) or "none"],
        ["  Temporal features",
         f"{len(groups['change'])} change, {len(groups['relchange'])} relative change, "
         f"{len(groups['rollstd'])} rolling std (windows 1h, 3h, 6h, 12h)"],
        ["Missing-value handling", "Median imputation, fitted on the training split only"],
        ["Class imbalance handling", "scale_pos_weight computed from the training split"],
        ["Selected configuration", str(config.get("name", "n/a"))],
    ]

    for key in ("n_estimators", "max_depth", "learning_rate", "min_child_weight",
                "subsample", "colsample_bytree", "gamma"):
        if key in config:
            rows.append([f"  {key}", str(config[key])])

    rows += [
        ["Training period", str(package.get("training_period", "n/a"))],
        ["Validation period", str(package.get("validation_period", "n/a"))],
        ["Test period", str(package.get("testing_period", "n/a"))],
        ["Split type", "Chronological (no shuffling across time)"],
        ["Random state", str(package.get("random_state", "n/a"))],
        ["Model file", str(MODEL_PATH.relative_to(PROJECT_ROOT))],
    ]

    story.append(_table(rows, [5.5 * cm, 11.5 * cm], S))

    # ---------------- 4. Data ----------------
    story.append(Paragraph("4. Data", S["h1"]))

    try:
        n_all = int(len(load_data()))
    except Exception:
        n_all = None

    rows = [
        ["Item", "Value"],
        ["Magnetic parameters", "SDO/HMI SHARP CEA series (hmi.sharp_cea_720s), 12-minute cadence"],
        ["Flare labels", "GOES M- and X-class flare start times, matched to active regions"],
        ["Feature dataset", str(FEATURES_PATH.relative_to(PROJECT_ROOT))],
        ["Rows in feature dataset", f"{n_all:,}" if n_all else "n/a"],
        ["Rows in test period", f"{ctx['n_test']:,}"],
        ["SHARP parameters available",
         "17 of the 25 standard parameters. TOTFX, TOTFY, TOTFZ, EPSX, EPSY, EPSZ, TOTBSQ and "
         "TOTABSTWIST returned 'Invalid KeyLink' for every record from this series and were excluded."],
    ]
    story.append(_table(rows, [5.5 * cm, 11.5 * cm], S))

    # ---------------- 5. Evaluation ----------------
    story.append(Paragraph("5. Evaluation on the held-out test period", S["h1"]))

    perf = ctx["perf"]
    xgb_conf = ctx["xgb_conf"]
    xgb_row = _perf_row(perf, "XGBoost")
    cnn_row = _perf_row(perf, "CNN")
    fus_row = _perf_row(perf, "Fusion")
    cnn = ctx["cnn"]
    fus = ctx["fusion"].get("metrics", {})

    rows = [["Model", "Evaluation set", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]]

    if xgb_row is not None:
        rows.append([
            "XGBoost", f"Full test period ({ctx['n_test']:,} rows)",
            _fmt(xgb_row["ROC-AUC"]), _fmt(xgb_row["PR-AUC"]),
            _fmt(xgb_conf["precision"]), _fmt(xgb_conf["recall"]), _fmt(xgb_conf["f1"]),
        ])

    if cnn_row is not None and cnn:
        rows.append([
            "CNN", "Test observations with a magnetogram image",
            _fmt(cnn_row["ROC-AUC"]), _fmt(cnn_row["PR-AUC"]),
            _fmt(cnn.get("test_precision")), _fmt(cnn.get("test_recall")), _fmt(cnn.get("test_f1")),
        ])

    if fus_row is not None:
        n_f = ctx["fusion"].get("n")
        rows.append([
            "Fusion", f"Same subset with both inputs (n={n_f:,})" if n_f else "Subset with both inputs",
            _fmt(fus_row["ROC-AUC"]), _fmt(fus_row["PR-AUC"]),
            _fmt(fus.get("precision")), _fmt(fus.get("recall")), _fmt(fus.get("f1")),
        ])

    story.append(_table(rows, [2.0 * cm, 5.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm], S))
    story.append(Paragraph(
        escape(
            "XGBoost is evaluated on every test-period observation; the CNN and fusion models can "
            "only score observations that have a magnetogram image, so those figures cover a much "
            "smaller subset and are not directly comparable. ROC-AUC measures how well a model ranks "
            "flare observations above non-flare ones; PR-AUC is more informative under strong class "
            "imbalance."
        ),
        S["small"],
    ))
    story.append(Spacer(1, 6))
    story.append(_img(chart_metric_bars(perf), 14))

    conf_items = [("XGBoost (full test period)", xgb_conf["confusion_matrix"])]

    if "cm" in ctx["fusion"]:
        conf_items.append(("Fusion (image subset)", ctx["fusion"]["cm"]))

    story.append(Spacer(1, 8))
    story.append(KeepTogether([
        _img(chart_confusions(conf_items), 15),
        Paragraph(escape("Confusion matrices at each model's saved decision threshold."), S["small"]),
    ]))

    # ---------------- 6. Model selection and importance ----------------
    story.append(Paragraph("6. Model selection and feature importance", S["h1"]))

    if HYPERPARAM_RESULTS_PATH.exists():

        hp = pd.read_csv(HYPERPARAM_RESULTS_PATH)
        rows = [["Configuration", "Validation ROC-AUC", "Validation PR-AUC", "Selected"]]

        for _, r in hp.iterrows():
            rows.append([
                r["model"], _fmt(r["validation_roc_auc"]), _fmt(r["validation_pr_auc"]),
                "yes" if r["model"] == config.get("name") else "",
            ])

        story.append(_table(rows, [5.5 * cm, 4 * cm, 4 * cm, 3.5 * cm], S))
        story.append(Paragraph(
            escape(
                "Validation scores differ only modestly between configurations, and the validation "
                "window is short (May-June 2012), so the choice of configuration is not strongly "
                "supported by the validation signal."
            ),
            S["small"],
        ))
        story.append(Spacer(1, 6))

    story.append(_img(chart_global_importance(predictor, 15), 15))

    # ---------------- 7. Limitations ----------------
    story.append(Paragraph("7. Limitations", S["h1"]))

    limitations = [
        "Single-year data: all training, validation and testing use 2012. Results may not generalise "
        "to other phases of the solar cycle.",
        "Weak validation signal: model selection used a two-month validation window and configuration "
        "differences are small (see section 6).",
        "Missing parameters: 8 of the 25 standard SHARP parameters were unavailable from the queried "
        "series and are not used.",
        "Uncalibrated scores: class weighting inflates predicted probabilities, so scores and risk "
        "bands are relative warning levels, not calibrated probabilities.",
        "Low precision and recall at the saved threshold: most alerts are false alarms and most flares "
        "are missed (see section 5).",
        "Subset evaluation: CNN and fusion metrics cover only observations with a magnetogram image and "
        "are not directly comparable to XGBoost's full-test metrics. The live score in this report "
        "comes from the tabular model only.",
        "Fusion adds little: a logistic regression on two probabilities, fitted on a small validation "
        "subset, does not meaningfully improve on the CNN alone.",
        "Label noise: flares without an attributed active region are excluded from labels, so some "
        "positive observations may be labelled negative.",
        "Autocorrelation: consecutive observations of the same active region are highly correlated, so "
        "the effective sample size is smaller than the row count. No confidence intervals are reported.",
        "Research prototype: not validated for operational use.",
    ]

    for text in limitations:
        story.append(Paragraph(escape(text), S["bullet"], bulletText="\u2022"))

    # ---------------- 8. Reproducibility ----------------
    story.append(Paragraph("8. Exploratory experiment: tabular Double Q-learning", S["h1"]))

    rl_path = PROJECT_ROOT / "results" / "rl_dry_run" / "rl_results.csv"
    rl_cfg_path = PROJECT_ROOT / "results" / "rl_dry_run" / "rl_config.json"

    if rl_path.exists():

        rl = pd.read_csv(rl_path)

        feature_text = ""
        if rl_cfg_path.exists():
            cfg = json.loads(rl_cfg_path.read_text(encoding="utf-8"))
            feature_text = " State features: " + ", ".join(cfg.get("state_features", [])) + "."

        story.append(Paragraph(
            escape(
                "A tabular Double Q-learning agent was trained on the real 2012 training months with the "
                "paper's asymmetric reward (true positive +5, true negative +1, false positive -1, missed "
                "flare -8 by default, with a sweep over harsher missed-flare penalties chosen on the "
                "validation period). Actions are alert / no alert; the state is a quantile-binned summary "
                "of the most important XGBoost features." + feature_text
            ),
            S["body"],
        ))

        rows = [["Policy", "Test TSS", "Test recall", "Test false-alarm rate", "Test precision", "Alerts"]]

        for _, r in rl.iterrows():
            rows.append([
                r["policy"], _fmt(r["test_tss"]), _fmt(r["test_recall"]),
                _fmt(r["test_far"]), _fmt(r["test_precision"]), f"{int(r['test_alerts']):,}",
            ])

        story.append(_table(rows, [5.6 * cm, 2.0 * cm, 2.2 * cm, 3.0 * cm, 2.4 * cm, 1.8 * cm], S))
        story.append(Paragraph(
            escape(
                "The reward depends only on the action and the true outcome, not on how the action changes "
                "later states, so the learned policy is equivalent to a cost-sensitive threshold on binned "
                "flare rates. This experiment shows the reward weighting working on real data; it does not "
                "show that reinforcement learning forecasts flares better than XGBoost."
            ),
            S["small"],
        ))

    else:
        story.append(Paragraph(
            escape("The reinforcement-learning experiment has not been run (scripts/rl_dry_run.py)."),
            S["body"],
        ))

    story.append(Paragraph("9. Reproducibility", S["h1"]))

    rows = [
        ["Item", "Value"],
        ["Python", sys.version.split()[0]],
        ["xgboost", _version("xgboost")],
        ["scikit-learn", _version("scikit-learn")],
        ["pandas", _version("pandas")],
        ["numpy", _version("numpy")],
        ["streamlit", _version("streamlit")],
        ["reportlab", _version("reportlab")],
        ["Report generated", ctx["generated"]],
    ]
    story.append(_table(rows, [5.5 * cm, 11.5 * cm], S))

    return _render(story, "Solar Flare Prediction: Technical Report")


# ============================================================
# MACHINE-READABLE EXPORT
# ============================================================

def build_technical_json(observation, history, selected_ar, selected_time):

    ctx = _collect(observation, history, selected_ar, selected_time)
    predictor = ctx["predictor"]
    package = predictor.model_package

    contributions = []

    if ctx["contrib"] is not None:
        for _, r in ctx["contrib"].head(25).iterrows():
            contributions.append({
                "feature": r["feature"],
                "value": None if pd.isna(r["value"]) else float(r["value"]),
                "imputed": bool(r["imputed"]),
                "contribution_logodds": float(r["contribution"]),
            })

    perf_records = ctx["perf"].to_dict(orient="records")

    payload = {
        "generated": ctx["generated"],
        "observation": {
            "noaa_ar": ctx["ar"],
            "harpnum": ctx["harp"],
            "time": ctx["time"],
            "observed_label": ctx["actual"],
        },
        "prediction": {
            "score": ctx["prob"],
            "risk_band": ctx["risk"],
            "threshold": ctx["threshold"],
            "predicted_class": ctx["pred_class"],
            "baseline_logodds": ctx["baseline"],
            "features_imputed": ctx["n_imputed"],
        },
        "contributions_top25": contributions,
        "model": {
            "n_features": len(predictor.features),
            "best_config": package.get("best_config"),
            "training_period": package.get("training_period"),
            "validation_period": package.get("validation_period"),
            "testing_period": package.get("testing_period"),
        },
        "performance": perf_records,
        "xgboost_test_confusion": {
            "matrix": ctx["xgb_conf"]["confusion_matrix"].tolist(),
            "precision": ctx["xgb_conf"]["precision"],
            "recall": ctx["xgb_conf"]["recall"],
            "f1": ctx["xgb_conf"]["f1"],
        },
    }

    return json.dumps(payload, indent=2, default=str).encode("utf-8")