r"""Compare logged live predictions against observed M/X flares.

Usage:
    python scripts\verify_live_log.py --flares data\live_log\recent_flares.csv

The flares CSV needs columns: start_time, flare_class, active_region
(the same format as data/raw/goes/goes_flares_2012.csv). Four-digit region
numbers get +10000 so they match SHARP's NOAA_AR numbering. The file is
assumed complete up to --covered-until (default: now, UTC).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "live_log" / "live_predictions.csv"
OUT_PATH = PROJECT_ROOT / "data" / "live_log" / "live_outcomes.csv"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--flares", required=True)
    parser.add_argument("--covered-until", default=None)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print("No live log yet. Click 'Fetch Latest Data' in Live Mode first.")
        sys.exit(0)

    log = pd.read_csv(LOG_PATH, parse_dates=["observation_time"])

    flares = pd.read_csv(args.flares)
    flares["start_time"] = pd.to_datetime(flares["start_time"], errors="coerce")
    flares["active_region"] = pd.to_numeric(flares["active_region"], errors="coerce")
    flares = flares.dropna(subset=["start_time", "active_region"])
    flares = flares[flares["flare_class"].astype(str).str.upper().str.startswith(("M", "X"))].copy()
    flares["NOAA_AR"] = flares["active_region"].astype(int)
    flares.loc[flares["NOAA_AR"] < 10000, "NOAA_AR"] += 10000

    covered = pd.Timestamp(args.covered_until) if args.covered_until else pd.Timestamp.now("UTC").tz_localize(None)

    matured = log[log["observation_time"] + pd.Timedelta(hours=24) <= covered].copy()

    print(f"Logged predictions: {len(log)} | matured (24h elapsed and covered): {len(matured)}")

    if len(matured) == 0:
        print("Nothing to verify yet.")
        sys.exit(0)

    outcomes = []

    for _, row in matured.iterrows():
        end = row["observation_time"] + pd.Timedelta(hours=24)
        hit = flares[
            (flares["NOAA_AR"] == row["NOAA_AR"])
            & (flares["start_time"] > row["observation_time"])
            & (flares["start_time"] <= end)
        ]
        outcomes.append(int(len(hit) > 0))

    matured["flare_within_24h"] = outcomes
    matured.to_csv(OUT_PATH, index=False)

    y = matured["flare_within_24h"].to_numpy()
    p = matured["xgb_probability"].to_numpy()
    pred = (p >= args.threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    print(f"Observed flares: {int(y.sum())} of {len(y)}")
    print(f"At threshold {args.threshold}: TP {tp}, FP {fp}, FN {fn}, TN {tn}")

    if len(np.unique(y)) > 1:
        print(f"XGBoost ROC-AUC: {roc_auc_score(y, p):.4f}")
    else:
        print("Only one outcome class so far, so ROC-AUC is not defined.")

    print("Small samples: treat these as anecdotes, not performance estimates.")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()