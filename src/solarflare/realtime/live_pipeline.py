"""Live solar flare prediction pipeline.

Pulls recent SHARP scalar data and a current magnetogram image
directly from JSOC, builds the same feature set the production
model was trained on, and runs XGBoost + CNN + the trained fusion
model.

Risk bands are taken from the XGBoost probability, matching the rest
of the dashboard. The CNN and fusion models were trained on a
50/50 class-balanced image sample, so their probabilities are on a
different scale and are shown as supporting signals only.
"""

import io
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"

for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.predict import SolarFlarePredictor  # noqa: E402

JSOC_BASE_URL = "http://jsoc.stanford.edu"

# Definitive series: fully calibrated but can lag real time by hours to over
# a day. NRT: near-real-time, lower latency, meant for live use. Try NRT
# first, fall back to the definitive series over a wider window.
PRIMARY_SERIES = "hmi.sharp_cea_720s_nrt"
FALLBACK_SERIES = "hmi.sharp_cea_720s"

PRIMARY_LOOKBACK_HOURS = 36
FALLBACK_LOOKBACK_HOURS = 72

STALE_HOURS = 3
MAGNETOGRAM_ATTEMPTS = 8

GOOD_BASE_FEATURES = [
    "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD",
    "TOTUSJZ", "MEANALP", "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP",
    "MEANPOT", "TOTPOT", "MEANSHR", "SHRGT45", "R_VALUE",
]

LOG_FEATURES = ["USFLUX", "TOTUSJH", "TOTPOT", "MEANPOT"]

TIME_WINDOWS = {"1h": 1, "3h": 3, "6h": 6, "12h": 12}
ROLL_WINDOWS = {"1h": pd.Timedelta(hours=1), "3h": pd.Timedelta(hours=3),
                "6h": pd.Timedelta(hours=6), "12h": pd.Timedelta(hours=12)}

XGB_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_2012_temporal_features_25params.joblib"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn_magnetogram.pt"
FUSION_MODEL_PATH = PROJECT_ROOT / "results" / "ensemble_25params" / "fusion_model.joblib"

_predictor = None
_cnn_model = None
_cnn_device = None
_fusion = None


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SolarFlarePredictor(str(XGB_MODEL_PATH))
    return _predictor


def get_cnn():
    global _cnn_model, _cnn_device
    if _cnn_model is None:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, 1))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(CNN_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _cnn_model, _cnn_device = model, device
    return _cnn_model, _cnn_device


def get_fusion():
    global _fusion
    if _fusion is None:
        _fusion = joblib.load(FUSION_MODEL_PATH) if FUSION_MODEL_PATH.exists() else None
    return _fusion


# ============================================================
# LIVE SHARP FETCH
# ============================================================

def _query_series(client, series, lookback_hours, keys):

    start = _utc_now() - timedelta(hours=lookback_hours)

    query = (
        f"{series}[][{start.strftime('%Y.%m.%d_%H:%M:%S')}_TAI/"
        f"{lookback_hours}h@12m]"
    )

    print(f"Querying {series} ({lookback_hours}h window): {query}")

    result = client.query(query, key=keys, skip_conversion=["QUALITY"])

    print(f"  -> {0 if result is None else len(result)} rows")

    return result


def fetch_live_sharp():
    import drms

    client = drms.Client()
    keys = ["T_REC", "HARPNUM", "NOAA_AR", "QUALITY"] + GOOD_BASE_FEATURES

    result = _query_series(client, PRIMARY_SERIES, PRIMARY_LOOKBACK_HOURS, keys)

    if result is not None and len(result) > 0:
        return result, PRIMARY_SERIES

    print(f"{PRIMARY_SERIES} returned nothing, falling back to {FALLBACK_SERIES}...")

    result = _query_series(client, FALLBACK_SERIES, FALLBACK_LOOKBACK_HOURS, keys)

    if result is not None and len(result) > 0:
        return result, FALLBACK_SERIES

    return None, None


def prepare_live_data(raw):

    data = raw.copy()

    data["observation_time"] = pd.to_datetime(
        data["T_REC"], format="%Y.%m.%d_%H:%M:%S_TAI", errors="coerce"
    )

    data["NOAA_AR"] = pd.to_numeric(data["NOAA_AR"], errors="coerce")

    data = data[data["NOAA_AR"].notna() & (data["NOAA_AR"] != 0)].copy()
    data = data.dropna(subset=["observation_time"])

    for col in GOOD_BASE_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.drop_duplicates(subset=["HARPNUM", "T_REC"])
    data = data.sort_values(["NOAA_AR", "observation_time"]).reset_index(drop=True)

    return data


# ============================================================
# TEMPORAL FEATURES (ported from scripts/engineer_temporal_features.py
# so live predictions use the exact same math as training)
# ============================================================

def _time_aware_lag(data, feature, hours, max_gap=pd.Timedelta(minutes=24)):

    result = pd.Series(np.nan, index=data.index, dtype=float)
    target_delta = pd.Timedelta(hours=hours)

    for ar, group in data.groupby("NOAA_AR", sort=False):

        group = group.sort_values("observation_time")
        times = group["observation_time"]
        values = group[feature]
        target_times = times - target_delta

        positions = times.searchsorted(target_times, side="right") - 1
        valid = positions >= 0

        lagged = np.full(len(group), np.nan, dtype=float)

        if valid.any():
            valid_positions = positions[valid]
            historical_times = times.iloc[valid_positions].to_numpy()
            requested_times = target_times.iloc[np.where(valid)[0]].to_numpy()
            gaps = requested_times - historical_times
            acceptable = gaps <= max_gap
            valid_indices = np.where(valid)[0]
            accepted_indices = valid_indices[acceptable]
            accepted_positions = valid_positions[acceptable]
            lagged[accepted_indices] = values.iloc[accepted_positions].to_numpy()

        result.loc[group.index] = lagged

    return result


def engineer_live_features(data):

    data = data.copy()

    for feature in LOG_FEATURES:
        data[f"LOG_{feature}"] = np.log1p(data[feature])

    data["observation_hour"] = data["observation_time"].dt.hour
    data["day_of_year"] = data["observation_time"].dt.dayofyear

    for feature in GOOD_BASE_FEATURES:
        for period, hours in TIME_WINDOWS.items():

            lagged = _time_aware_lag(data, feature, hours)

            data[f"{feature}_CHANGE_{period}"] = data[feature] - lagged

            denom = lagged.abs()
            data[f"{feature}_RELCHANGE_{period}"] = np.where(
                denom > 0, (data[feature] - lagged) / denom, np.nan
            )

    for feature in GOOD_BASE_FEATURES:
        for period, delta in ROLL_WINDOWS.items():

            result = pd.Series(np.nan, index=data.index, dtype=float)

            for region, group in data.groupby("NOAA_AR", sort=False):

                group = group.sort_values("observation_time")
                series = group.set_index("observation_time")[feature]
                rolling = series.shift(1).rolling(delta, min_periods=3).std()
                result.loc[group.index] = rolling.to_numpy()

            data[f"{feature}_ROLLSTD_{period}"] = result

    return data


def get_latest_per_region(data, stale_hours=STALE_HOURS):

    idx = data.groupby("NOAA_AR")["observation_time"].idxmax()
    latest = data.loc[idx].reset_index(drop=True)

    newest = latest["observation_time"].max()
    cutoff = newest - pd.Timedelta(hours=stale_hours)

    return latest[latest["observation_time"] >= cutoff].reset_index(drop=True)


# ============================================================
# LIVE MAGNETOGRAM
# ============================================================

def _read_br_image(content):
    """Parse FITS bytes into a normalised 224x224 float32 image."""

    from astropy.io import fits

    hdul = fits.open(io.BytesIO(content))
    img = hdul[1].data if len(hdul) > 1 else hdul[0].data
    hdul.close()

    if img is None:
        return None

    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, -3000, 3000) / 3000.0

    th, tw = 224, 224
    h, w = img.shape

    if h < th or w < tw:
        pad_h, pad_w = max(0, th - h), max(0, tw - w)
        img = np.pad(
            img,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
        )
        h, w = img.shape

    y0, x0 = (h - th) // 2, (w - tw) // 2

    return img[y0:y0 + th, x0:x0 + tw]


def fetch_live_magnetogram(client, series, harpnum, t_recs):
    """Try the region's most recent timestamps until one image downloads.

    Returns (image, t_rec_used, reason). reason is empty on success.
    """

    reasons = []

    for t_rec in t_recs:

        try:
            query = f"{series}[{int(harpnum)}][{t_rec}]"
            keys, segs = client.query(query, key="T_REC", seg="Br")
        except Exception as exc:
            reasons.append(f"query failed ({type(exc).__name__})")
            continue

        if segs is None or len(segs) == 0 or "Br" not in segs.columns:
            reasons.append("no Br segment listed")
            continue

        seg_path = str(segs.iloc[0]["Br"]).strip()

        if not seg_path or seg_path.lower() in ("nan", "none"):
            reasons.append("Br file not available yet")
            continue

        url = seg_path if seg_path.startswith("http") else JSOC_BASE_URL + seg_path

        try:
            resp = requests.get(url, timeout=90)
        except Exception as exc:
            reasons.append(f"download failed ({type(exc).__name__})")
            continue

        if resp.status_code != 200 or len(resp.content) < 1000:
            reasons.append(f"download returned HTTP {resp.status_code}")
            continue

        try:
            img = _read_br_image(resp.content)
        except ImportError:
            return None, None, "astropy is not installed (run: pip install astropy)"
        except Exception as exc:
            reasons.append(f"could not read FITS ({type(exc).__name__})")
            continue

        if img is not None:
            return img, t_rec, ""

        reasons.append("empty image")

    summary = "; ".join(dict.fromkeys(reasons)) or "no attempts made"

    return None, None, summary


def predict_cnn(img):

    import torch

    model, device = get_cnn()

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().item()

    return float(prob)


# ============================================================
# ORCHESTRATOR
# ============================================================

def run_live_predictions():

    import drms

    predictor = get_predictor()
    fusion = get_fusion()

    raw, series_used = fetch_live_sharp()

    if raw is None or len(raw) == 0:
        return pd.DataFrame(), (
            "No live SHARP data returned from either the near-real-time or "
            "definitive JSOC series. JSOC may be experiencing a processing "
            "delay right now -- try again in a while."
        )

    data = prepare_live_data(raw)

    if len(data) == 0:
        return pd.DataFrame(), (
            "No usable live observations after cleaning "
            "(no NOAA-numbered active regions in range)."
        )

    engineered = engineer_live_features(data)
    latest = get_latest_per_region(engineered)

    client = drms.Client()

    rows = []

    for i, row in latest.iterrows():

        obs = latest.loc[[i]]

        missing = [f for f in predictor.features if f not in obs.columns]
        if missing:
            continue

        xgb_prob = float(predictor.predict(obs).iloc[0]["flare_probability"])

        region_obs = data[data["NOAA_AR"] == row["NOAA_AR"]].sort_values(
            "observation_time", ascending=False
        )
        t_recs = region_obs["T_REC"].drop_duplicates().head(MAGNETOGRAM_ATTEMPTS).tolist()

        img, t_rec_used, reason = fetch_live_magnetogram(
            client, series_used, row["HARPNUM"], t_recs
        )

        cnn_prob = float("nan")
        fusion_prob = float("nan")
        image_time = pd.NaT
        note = ""

        if img is not None:

            image_time = pd.to_datetime(
                t_rec_used, format="%Y.%m.%d_%H:%M:%S_TAI", errors="coerce"
            )

            try:
                cnn_prob = predict_cnn(img)

                if fusion is not None:
                    X = np.array([[xgb_prob, cnn_prob]])
                    fusion_prob = float(fusion["model"].predict_proba(X)[:, 1][0])

            except Exception as exc:
                note = f"CNN scoring failed ({type(exc).__name__})"

        else:
            note = f"No live magnetogram ({reason}); XGBoost-only score"

        rows.append({
            "NOAA_AR": int(row["NOAA_AR"]),
            "HARPNUM": int(row["HARPNUM"]),
            "observation_time": row["observation_time"],
            "data_series": series_used,
            "xgb_probability": xgb_prob,
            "cnn_probability": cnn_prob,
            "fusion_probability": fusion_prob,
            "image_time": image_time,
            "risk_level": predictor.classify_risk(xgb_prob),
            "note": note,
        })

    results = pd.DataFrame(rows)

    if len(results) > 0:
        results = results.sort_values("xgb_probability", ascending=False).reset_index(drop=True)

    return results, None

# ============================================================
# LIVE RUN WITH PER-REGION CONTEXT (used by the dashboard reports)
# ============================================================

def run_live_predictions_with_context():
    """Like run_live_predictions, but also returns per-region context.

    Returns (results, contexts, error). contexts maps NOAA_AR to a dict with the
    one-row feature frame, the region's recent history, the magnetogram image (or
    None) and the individual model probabilities, so reports can be built later.
    """

    import drms

    predictor = get_predictor()
    fusion = get_fusion()

    raw, series_used = fetch_live_sharp()

    if raw is None or len(raw) == 0:
        return pd.DataFrame(), {}, (
            "No live SHARP data returned from either the near-real-time or "
            "definitive JSOC series. JSOC may be experiencing a processing "
            "delay right now -- try again in a while."
        )

    data = prepare_live_data(raw)

    if len(data) == 0:
        return pd.DataFrame(), {}, (
            "No usable live observations after cleaning "
            "(no NOAA-numbered active regions in range)."
        )

    engineered = engineer_live_features(data)
    latest = get_latest_per_region(engineered)

    client = drms.Client()

    rows = []
    contexts = {}

    for i, row in latest.iterrows():

        obs = latest.loc[[i]]

        missing = [f for f in predictor.features if f not in obs.columns]
        if missing:
            continue

        xgb_prob = float(predictor.predict(obs).iloc[0]["flare_probability"])

        region_obs = data[data["NOAA_AR"] == row["NOAA_AR"]].sort_values(
            "observation_time", ascending=False
        )
        t_recs = region_obs["T_REC"].drop_duplicates().head(MAGNETOGRAM_ATTEMPTS).tolist()

        img, t_rec_used, reason = fetch_live_magnetogram(
            client, series_used, row["HARPNUM"], t_recs
        )

        cnn_prob = float("nan")
        fusion_prob = float("nan")
        image_time = pd.NaT
        note = ""

        if img is not None:

            image_time = pd.to_datetime(
                t_rec_used, format="%Y.%m.%d_%H:%M:%S_TAI", errors="coerce"
            )

            try:
                cnn_prob = predict_cnn(img)

                if fusion is not None:
                    X = np.array([[xgb_prob, cnn_prob]])
                    fusion_prob = float(fusion["model"].predict_proba(X)[:, 1][0])

            except Exception as exc:
                note = f"CNN scoring failed ({type(exc).__name__})"

        else:
            note = f"No live magnetogram ({reason}); XGBoost-only score"

        ar = int(row["NOAA_AR"])

        rows.append({
            "NOAA_AR": ar,
            "HARPNUM": int(row["HARPNUM"]),
            "observation_time": row["observation_time"],
            "data_series": series_used,
            "xgb_probability": xgb_prob,
            "cnn_probability": cnn_prob,
            "fusion_probability": fusion_prob,
            "image_time": image_time,
            "risk_level": predictor.classify_risk(xgb_prob),
            "note": note,
        })

        history = (
            engineered[engineered["NOAA_AR"] == row["NOAA_AR"]]
            .sort_values("observation_time")
            .reset_index(drop=True)
        )

        contexts[ar] = {
            "observation": obs.copy(),
            "history": history,
            "image": img,
            "series": series_used,
            "image_time": image_time,
            "xgb_probability": xgb_prob,
            "cnn_probability": cnn_prob,
            "fusion_probability": fusion_prob,
            "note": note,
        }

    results = pd.DataFrame(rows)

    if len(results) > 0:
        results = results.sort_values("xgb_probability", ascending=False).reset_index(drop=True)

    return results, contexts, None