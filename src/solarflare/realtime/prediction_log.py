"""Append-only log of live predictions, used for later verification."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = PROJECT_ROOT / "data" / "live_log" / "live_predictions.csv"


def log_predictions(results):
    """Append a live run to the log; never raises, returns rows logged."""

    try:
        if results is None or len(results) == 0:
            return 0

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        new = results.copy()
        new.insert(0, "logged_at_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

        if LOG_PATH.exists():
            combined = pd.concat([pd.read_csv(LOG_PATH), new], ignore_index=True)
        else:
            combined = new

        combined["observation_time"] = pd.to_datetime(combined["observation_time"])
        combined = combined.drop_duplicates(subset=["NOAA_AR", "observation_time"], keep="last")
        combined.to_csv(LOG_PATH, index=False)

        return len(new)

    except Exception:
        return 0