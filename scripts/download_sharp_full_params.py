from pathlib import Path
import time
import pandas as pd
import drms

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("data/raw/sharp/daily_full_params")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES = "hmi.sharp_cea_720s"
CADENCE = "12m"

# Full Bobra & Couvidat (2015) 25-parameter SHARP feature set,
# plus the metadata/identifier columns the rest of the pipeline expects.
KEYS = [
    "T_REC",
    "HARPNUM",
    "NOAA_AR",
    "QUALITY",
    "MASK",
    "CMASK",
    # 25 physical SHARP parameters
    "USFLUX",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZD",
    "TOTUSJZ",
    "MEANALP",
    "MEANJZH",
    "TOTUSJH",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "TOTPOT",
    "MEANSHR",
    "SHRGT45",
    "R_VALUE",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
    "TOTBSQ",
    "TOTABSTWIST",
]

# One query per day for the whole of 2012 (adjust range if needed)
DAYS = pd.date_range("2012-01-01", "2012-12-31", freq="D")

# Politeness delay between successful queries, seconds
REQUEST_DELAY = 2


def query_day(client, day):

    query = (
        f"{SERIES}[]"
        f"[{day.strftime('%Y.%m.%d')}_00:00:00_TAI/1d@{CADENCE}]"
    )

    print(f"Querying: {query}")

    result = client.query(
        query,
        key=KEYS,
        skip_conversion=["QUALITY"],
    )

    return result


def main():

    client = drms.Client()

    print("=" * 60)
    print(" DOWNLOADING FULL 25-PARAMETER SHARP DATA (2012, daily chunks)")
    print("=" * 60)

    total_days = len(DAYS)

    for i, day in enumerate(DAYS, start=1):

        day_label = day.strftime("%Y_%m_%d")
        output_path = OUTPUT_DIR / f"sharp_{day_label}_full_params.parquet"

        if output_path.exists():
            print(f"[SKIP] ({i}/{total_days}) Already downloaded: {output_path}")
            continue

        print()
        print(f"----- ({i}/{total_days}) {day_label} -----")

        attempts = 0
        result = None

        while attempts < 3:

            try:
                result = query_day(client, day)
                break

            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")
                time.sleep(10)

        if result is None:
            print(f"[FAILED] Giving up on {day_label} after 3 attempts.")
            continue

        print(f"Rows returned: {len(result)}")

        if len(result) > 0:
            result.to_parquet(output_path, index=False)
            print(f"Saved: {output_path}")
        else:
            print(f"[SKIP] No data for {day_label}, not saving empty file.")

        time.sleep(REQUEST_DELAY)

    print()
    print("=" * 60)
    print(" DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()