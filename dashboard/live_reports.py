"""PDF / JSON reports for Live Mode observations.

Reuses the helpers in reports.py so live reports look and read like the
historical ones, and adds the live-specific context: data source, data age,
the magnetogram, the CNN / fusion signals and the live-data caveats.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from common import MODEL_PATH, PROJECT_ROOT, format_value  # noqa: E402
from reports import (  # noqa: E402
    DAILY_IMPACT,
    GLOSSARY,
    WHAT_THE_MODEL_SAW,
    _collect,
    _feature_groups,
    _fmt,
    _img,
    _ratio_text,
    _render,
    _risk_box,
    _score_text,
    _styles,
    _table,
    _version,
    chart_contributions,
    chart_metric_bars,
    chart_risk_ladder,
    chart_temporal,
    describe_feature,
)
from solarflare.realtime import live_pipeline as lp  # noqa: E402


LIVE_PLAIN_BANNER = (
    "LIVE DATA -- this report uses the Sun's most recent measurements, not historical records. "
    "Nobody can check the score against what actually happens until the next 24 hours have "
    "passed, and the system has only been tested on data from 2012. Treat it as experimental."
)

LIVE_TECH_BANNER = (
    "LIVE DATA -- scored from JSOC near-real-time SHARP records. The models were trained and "
    "evaluated only on the definitive 2012 series; live scores have not been validated against "
    "outcomes, and the historical metrics in section 6 do not apply directly to this score."
)

LIVE_GLOSSARY_EXTRA = [
    ("Near-real-time data", "Measurements delivered within hours of being taken, before they go through the final calibration used for archived data."),
    ("Live score", "A score computed from the Sun's most recent measurements. Nobody can check it against what actually happened until the next 24 hours have passed."),
]

LIVE_LIMITATIONS = [
    "No live validation: this score cannot be compared with what actually happens until the following "
    "24 hours have passed, and no live validation has been performed.",
    "Near-real-time data: live records come from the near-real-time SHARP series, while the models were "
    "trained on the definitive series, so small calibration differences are possible.",
    "Distribution shift: training and testing used 2012, near the maximum of solar cycle 24; current "
    "conditions may differ.",
    "Image-model scale: the CNN and fusion models were trained on a class-balanced image sample (half "
    "flare, half no-flare), so their probabilities are not on the same scale as the XGBoost score and "
    "are not used for the risk band.",
    "Single image: the CNN sees one recent radial-field magnetogram, which can be slightly older than "
    "the latest tabular record.",
    "Region coverage: only NOAA-numbered active regions are scored, matching how the training labels "
    "were built.",
    "Uncalibrated scores: class weighting inflates predicted probabilities, so scores and risk bands "
    "are relative warning levels, not calibrated probabilities.",
    "Low precision and recall at the saved threshold: most alerts are false alarms and most flares are "
    "missed on the 2012 test period.",
    "Missing parameters: 8 of the 25 standard SHARP parameters were unavailable from the queried series "
    "and are not used.",
    "Research prototype: not validated for operational use.",
]


# ============================================================
# SMALL HELPERS
# ============================================================

def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _age_text(ts):
    try:
        hours = (_utc_now() - pd.Timestamp(ts).to_pydatetime()).total_seconds() / 3600
        return f"{hours:.1f} hours before this report"
    except Exception:
        return "age unknown"


def _prob_text(value):
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return _score_text(float(value))
    except (TypeError, ValueError):
        return "n/a"


def _num(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ts_text(ts):
    try:
        if ts is None or pd.isna(ts):
            return "none available"
        return f"{ts} UTC"
    except (TypeError, ValueError):
        return "none available"


def _live_banner(S, text):

    table = Table([[Paragraph(escape(text), S["body"])]], colWidths=[17 * cm])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#e0a800")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))

    return table


def chart_magnetogram(img):

    fig = Figure(figsize=(4.2, 4.2))
    ax = fig.subplots()

    im = ax.imshow(img, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Radial field / 3000 G", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return fig


# ============================================================
# PLAIN-LANGUAGE LIVE REPORT
# ============================================================

def build_live_plain_report(context, selected_ar, selected_time):

    ctx = _collect(context["observation"], context["history"], selected_ar, selected_time)
    S = _styles()
    story = []

    story.append(Paragraph("Live Solar Flare Risk Summary", S["title"]))
    story.append(Paragraph(
        escape(f"Active region NOAA {ctx['ar']}  |  Latest measurement {ctx['time']} UTC "
               f"({_age_text(ctx['time'])})  |  Report generated {ctx['generated']}"),
        S["small"],
    ))
    story.append(Spacer(1, 8))
    story.append(_live_banner(S, LIVE_PLAIN_BANNER))
    story.append(Spacer(1, 10))

    story.append(_risk_box(ctx, S))
    story.append(Spacer(1, 6))
    story.append(_img(chart_risk_ladder(ctx["prob"]), 15))

    cnn = context.get("cnn_probability")
    has_image = context.get("image") is not None

    if has_image and _num(cnn) is not None:
        image_sentence = (
            f"A second model that studies the region's magnetic image was also run and gave "
            f"{_prob_text(cnn)}. Its percentage reads much higher than the main score because it was "
            f"trained on a sample that was half flare and half quiet, so it is shown for reference "
            f"only and does not set the risk level."
        )
    else:
        image_sentence = (
            "No recent magnetic image was available for this region, so only the measurement-based "
            "model was used."
        )

    story.append(Paragraph("The short version", S["h1"]))
    story.append(Paragraph(
        escape(
            f"For this active region right now, the model's risk score is {_score_text(ctx['prob'])}, "
            f"placing it in the {ctx['risk'].lower()} band. {WHAT_THE_MODEL_SAW.get(ctx['risk'], '')} "
            f"{image_sentence} Terms in this report are explained in the glossary at the end."
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

        story.append(Paragraph(
            escape("A breakdown for this specific reading was not available."),
            S["body"],
        ))

    if has_image:
        story.append(Spacer(1, 6))
        story.append(KeepTogether([
            _img(chart_magnetogram(context["image"]), 7.5),
            Paragraph(
                escape(
                    f"The magnetic image of this region (taken {_ts_text(context.get('image_time'))}) that the "
                    f"second model studied. Red and blue show opposite magnetic polarity."
                ),
                S["small"],
            ),
        ]))

    temporal = chart_temporal(ctx["history"])

    if temporal is not None:
        story.append(Spacer(1, 6))
        story.append(KeepTogether([
            _img(temporal, 15),
            Paragraph(
                escape("How four key measurements of this region changed over the last day and a half "
                       "(each rescaled to be comparable)."),
                S["small"],
            ),
        ]))

    story.append(Paragraph("How much should you trust this?", S["h1"]))

    precision = ctx["xgb_conf"]["precision"]
    recall = ctx["xgb_conf"]["recall"]

    story.append(Paragraph(
        escape(
            f"This tool is a research prototype, tested on one year of historical data (2012). In that test, "
            f"{_ratio_text(precision)} of its alerts were followed by a real flare, and out of all the real "
            f"flares that year it caught {_ratio_text(recall)} of them."
        ),
        S["body"],
    ))
    story.append(Paragraph(
        escape(
            "Because this report uses live data, none of it has been checked against what actually "
            "happened. The measurements also arrive within hours and have not had the final calibration "
            "the model was trained on, so scores may be slightly off. Treat the result as one extra signal, "
            "not a forecast to rely on by itself: most alerts will not lead to a flare, and many flares "
            "will happen without an alert."
        ),
        S["body"],
    ))

    story.append(Paragraph("What is a solar flare?", S["h1"]))
    story.append(Paragraph(
        escape(
            "A solar flare is a sudden release of magnetic energy from the Sun's surface. The strongest ones "
            "(M-class and X-class) can disturb radio communication and GPS accuracy, affect satellites, and "
            "in extreme cases stress power grids or raise radiation exposure for high-altitude flights. "
            "Forecasting them is difficult, which is why this tool reports a risk level rather than a "
            "certainty."
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

    glossary_rows = [["Term", "Meaning"]] + [list(pair) for pair in GLOSSARY + LIVE_GLOSSARY_EXTRA]
    story.append(_table(glossary_rows, [4.5 * cm, 12.5 * cm], S))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        escape("This report is produced by a research prototype and must not be used as the sole "
               "basis for any operational decision."),
        S["small"],
    ))

    return _render(story, "Live Solar Flare Risk Summary")


# ============================================================
# TECHNICAL LIVE REPORT
# ============================================================

def build_live_technical_report(context, selected_ar, selected_time):

    ctx = _collect(context["observation"], context["history"], selected_ar, selected_time)
    predictor = ctx["predictor"]
    package = predictor.model_package
    S = _styles()
    story = []

    full_w = 17 * cm
    has_image = context.get("image") is not None

    story.append(Paragraph("Live Solar Flare Prediction: Technical Report", S["title"]))
    story.append(Paragraph(
        escape(f"Active region NOAA {ctx['ar']}  |  Latest record {ctx['time']} UTC  |  "
               f"Generated {ctx['generated']}"),
        S["small"],
    ))
    story.append(Spacer(1, 6))
    story.append(_live_banner(S, LIVE_TECH_BANNER))

    # ---------------- 1. Live observation summary ----------------
    story.append(Paragraph("1. Live observation summary", S["h1"]))

    rows = [
        ["Item", "Value"],
        ["NOAA active region", ctx["ar"]],
        ["HARP number", ctx["harp"] or "n/a"],
        ["Data series", str(context.get("series", "n/a"))],
        ["Latest SHARP record (UTC)", ctx["time"]],
        ["Age at report generation", _age_text(ctx["time"])],
        ["Magnetogram used for the CNN", _ts_text(context.get("image_time")) if has_image else "none available"],
        ["XGBoost score (basis of the risk band)", _score_text(ctx["prob"])],
        ["Risk band", ctx["risk"]],
        ["Decision threshold (saved with model)", _fmt(ctx["threshold"], 3)],
        ["Predicted class at threshold", "Flare (1)" if ctx["pred_class"] == 1 else "No flare (0)"],
        ["CNN probability", _prob_text(context.get("cnn_probability"))],
        ["Fusion probability", _prob_text(context.get("fusion_probability"))],
        ["Features imputed for this record", f"{ctx['n_imputed']} of {len(predictor.features)}"],
    ]
    story.append(_table(rows, [7.5 * cm, 9.5 * cm], S))
    story.append(Paragraph(
        escape(
            "The CNN and fusion models were trained on a class-balanced image sample (half flare, half "
            "no-flare), so their probabilities sit on a different scale from XGBoost's and are shown for "
            "reference only. The risk band uses the XGBoost score, matching the Predictor page."
        ),
        S["small"],
    ))
    story.append(Spacer(1, 6))
    story.append(_img(chart_risk_ladder(ctx["prob"]), 14))

    # ---------------- 2. Per-observation explanation ----------------
    story.append(Paragraph("2. Per-observation explanation", S["h1"]))

    if ctx["contrib"] is not None:

        story.append(Paragraph(
            escape(
                "Contributions are XGBoost's native TreeSHAP values in log-odds units for this record. "
                "Positive values raise the model score, negative values lower it. Baseline plus all "
                "contributions gives this record's final log-odds."
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
            escape("Per-observation contributions could not be computed for this record."),
            S["body"],
        ))

    story.append(PageBreak())

    # ---------------- 3. Inputs ----------------
    story.append(Paragraph("3. Inputs: magnetogram and recent evolution", S["h1"]))

    if has_image:
        story.append(KeepTogether([
            _img(chart_magnetogram(context["image"]), 8),
            Paragraph(
                escape("Radial magnetic field (Br) magnetogram scored by the CNN, clipped to +/-3000 G and "
                       "scaled to [-1, 1]."),
                S["small"],
            ),
        ]))
    else:
        story.append(Paragraph(
            escape(f"No magnetogram was available. {context.get('note', '')}"),
            S["body"],
        ))

    temporal = chart_temporal(ctx["history"])

    if temporal is not None:
        story.append(Spacer(1, 8))
        story.append(KeepTogether([
            _img(temporal, 15),
            Paragraph(
                escape("Standardised evolution of selected parameters for this active region over the "
                       "live query window."),
                S["small"],
            ),
        ]))

    # ---------------- 4. Live data pipeline ----------------
    story.append(Paragraph("4. Live data pipeline", S["h1"]))

    rows = [
        ["Item", "Value"],
        ["Data series used", str(context.get("series", "n/a"))],
        ["Preferred series", f"{lp.PRIMARY_SERIES} ({lp.PRIMARY_LOOKBACK_HOURS} h window, 12-minute cadence)"],
        ["Fallback series", f"{lp.FALLBACK_SERIES} ({lp.FALLBACK_LOOKBACK_HOURS} h window), used only if the preferred series returns nothing"],
        ["Region selection", f"NOAA-numbered regions only; regions whose latest record is more than {lp.STALE_HOURS} h older than the newest record are skipped"],
        ["Features", "Same code path as training: log features, calendar features, and 1/3/6/12-hour change, relative change and rolling standard deviation (time-aware lags, maximum 24-minute gap)"],
        ["Magnetogram", f"Br segment, clipped to +/-3000 G and scaled to [-1, 1], centre-cropped or padded to 224 x 224; up to {lp.MAGNETOGRAM_ATTEMPTS} most recent timestamps tried per region"],
        ["Risk band", "From the XGBoost score using the production bands: under 1%, 1-5%, 5-20%, 20% and above"],
        ["CNN and fusion", "ResNet18 on the magnetogram; fusion is a logistic regression on the XGBoost and CNN probabilities"],
    ]
    story.append(_table(rows, [4.5 * cm, 12.5 * cm], S))

    # ---------------- 5. Model card ----------------
    story.append(Paragraph("5. Model card: XGBoost tabular classifier", S["h1"]))

    groups = _feature_groups(predictor.features)
    config = package.get("best_config", {}) or {}

    rows = [
        ["Item", "Value"],
        ["Model type", "XGBClassifier (gradient-boosted decision trees), objective binary:logistic"],
        ["Prediction target", "M- or X-class flare starting within 24 hours of the observation, per active region"],
        ["Total features", str(len(predictor.features))],
        ["Base SHARP parameters", f"{len(groups['base'])}: " + ", ".join(groups["base"])],
        ["Selected configuration", str(config.get("name", "n/a"))],
        ["Training period", str(package.get("training_period", "n/a"))],
        ["Validation period", str(package.get("validation_period", "n/a"))],
        ["Test period", str(package.get("testing_period", "n/a"))],
        ["Model file", str(MODEL_PATH.relative_to(PROJECT_ROOT))],
    ]
    story.append(_table(rows, [5.5 * cm, 11.5 * cm], S))

    # ---------------- 6. Historical evaluation ----------------
    story.append(Paragraph("6. Historical evaluation (2012 test period)", S["h1"]))

    perf = ctx["perf"]
    conf = ctx["xgb_conf"]

    rows = [["Model", "ROC-AUC", "PR-AUC"]]
    for _, r in perf.iterrows():
        rows.append([r["Model"], _fmt(r["ROC-AUC"]), _fmt(r["PR-AUC"])])

    story.append(_table(rows, [7 * cm, 4 * cm, 4 * cm], S))
    story.append(Paragraph(
        escape(
            f"XGBoost at its saved threshold on the 2012 test period: precision {_fmt(conf['precision'])}, "
            f"recall {_fmt(conf['recall'])}, F1 {_fmt(conf['f1'])}. These figures describe historical "
            f"performance on the definitive 2012 series; they do not validate live scores. CNN and fusion "
            f"metrics cover only observations with a magnetogram image and are not directly comparable to "
            f"XGBoost's full-test metrics."
        ),
        S["small"],
    ))
    story.append(Spacer(1, 6))
    story.append(_img(chart_metric_bars(perf), 14))

    # ---------------- 7. Limitations ----------------
    story.append(Paragraph("7. Limitations", S["h1"]))

    for text in LIVE_LIMITATIONS:
        story.append(Paragraph(escape(text), S["bullet"], bulletText="\u2022"))

    # ---------------- 8. Reproducibility ----------------
    story.append(Paragraph("8. Reproducibility", S["h1"]))

    rows = [
        ["Item", "Value"],
        ["Python", sys.version.split()[0]],
        ["xgboost", _version("xgboost")],
        ["scikit-learn", _version("scikit-learn")],
        ["pandas", _version("pandas")],
        ["numpy", _version("numpy")],
        ["torch", _version("torch")],
        ["astropy", _version("astropy")],
        ["streamlit", _version("streamlit")],
        ["reportlab", _version("reportlab")],
        ["Report generated", ctx["generated"]],
    ]
    story.append(_table(rows, [5.5 * cm, 11.5 * cm], S))

    return _render(story, "Live Solar Flare Prediction: Technical Report")


# ============================================================
# MACHINE-READABLE EXPORT
# ============================================================

def build_live_json(context, selected_ar, selected_time):

    ctx = _collect(context["observation"], context["history"], selected_ar, selected_time)
    predictor = ctx["predictor"]

    contributions = []

    if ctx["contrib"] is not None:
        for _, r in ctx["contrib"].head(25).iterrows():
            contributions.append({
                "feature": r["feature"],
                "value": _num(r["value"]),
                "imputed": bool(r["imputed"]),
                "contribution_logodds": float(r["contribution"]),
            })

    payload = {
        "generated": ctx["generated"],
        "live": True,
        "observation": {
            "noaa_ar": ctx["ar"],
            "harpnum": ctx["harp"],
            "latest_record_utc": ctx["time"],
            "data_series": context.get("series"),
            "magnetogram_time_utc": None if context.get("image") is None else str(context.get("image_time")),
        },
        "prediction": {
            "xgboost_probability": ctx["prob"],
            "cnn_probability": _num(context.get("cnn_probability")),
            "fusion_probability": _num(context.get("fusion_probability")),
            "risk_band": ctx["risk"],
            "risk_basis": "xgboost",
            "threshold": ctx["threshold"],
            "predicted_class": ctx["pred_class"],
            "baseline_logodds": ctx["baseline"],
            "features_imputed": ctx["n_imputed"],
        },
        "model": {
            "n_features": len(predictor.features),
            "best_config": predictor.model_package.get("best_config"),
        },
        "contributions_top25": contributions,
        "caveats": LIVE_LIMITATIONS,
    }

    return json.dumps(payload, indent=2, default=str).encode("utf-8")