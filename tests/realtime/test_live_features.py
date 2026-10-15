from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from solarflare.realtime import live_pipeline as lp


def make_frame(hours=14, ar=11111):
    times = pd.date_range("2026-01-01 00:00:00", periods=int(hours * 5) + 1, freq="12min")
    rng = np.random.default_rng(0)

    data = pd.DataFrame({
        "NOAA_AR": ar,
        "HARPNUM": 1,
        "observation_time": times,
        "T_REC": [t.strftime("%Y.%m.%d_%H:%M:%S_TAI") for t in times],
    })

    for feature in lp.GOOD_BASE_FEATURES:
        data[feature] = rng.uniform(1.0, 100.0, len(data))

    return data


def test_one_hour_change_matches_five_steps_back():
    df = lp.engineer_live_features(make_frame())
    expected = df["USFLUX"].iloc[10] - df["USFLUX"].iloc[5]

    assert np.isclose(df["USFLUX_CHANGE_1h"].iloc[10], expected)
    assert df["USFLUX_CHANGE_1h"].iloc[:5].isna().all()


def test_stale_regions_are_dropped():
    fresh = make_frame(ar=11111)
    old = make_frame(ar=22222)
    old["observation_time"] = old["observation_time"] - pd.Timedelta(hours=10)

    latest = lp.get_latest_per_region(pd.concat([fresh, old], ignore_index=True))

    assert list(latest["NOAA_AR"]) == [11111]


@pytest.mark.skipif(not Path(lp.XGB_MODEL_PATH).exists(), reason="model file not present")
def test_live_features_cover_every_model_feature():
    package = joblib.load(lp.XGB_MODEL_PATH)
    engineered = lp.engineer_live_features(make_frame())

    missing = [f for f in package["features"] if f not in engineered.columns]

    assert not missing