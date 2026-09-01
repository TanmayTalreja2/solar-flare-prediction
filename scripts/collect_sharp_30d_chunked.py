import drms
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = datetime(2012, 4, 1)
END_DATE = datetime(2013, 1, 1)

CHUNK_DAYS = 7

OUTPUT_PATH = Path(
    "data/raw/sharp/sharp_2012_04_01_2012_12_31.parquet"
)


# ============================================================
# SHARP PARAMETERS
# ============================================================

KEYS = [
    "T_REC",
    "HARPNUM",
    "NOAA_AR",
    "QUALITY",
    "MASK",
    "CMASK",
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]


# ============================================================
# QUERY ONE CHUNK
# ============================================================

def query_chunk(
    client: drms.Client,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:

    duration = end - start

    query = (
        "hmi.sharp_cea_720s"
        f"[][{start.strftime('%Y.%m.%d_%H:%M:%S')}_TAI/"
        f"{duration.days}d@12m]"
    )

    print()
    print("----------------------------------------")
    print("Querying SHARP chunk")
    print(f"Start: {start}")
    print(f"End:   {end}")
    print(f"Duration: {duration.days} days")
    print("----------------------------------------")

    result = client.query(
        query,
        key=KEYS,
        skip_conversion=["QUALITY"],
    )

    print(
        f"Rows returned: {len(result)}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("========================================")
    print(" SHARP CHUNKED COLLECTION")
    print("========================================")

    print(
        f"Start date: {START_DATE}"
    )

    print(
        f"End date:   {END_DATE}"
    )

    print(
        f"Chunk size: {CHUNK_DAYS} days"
    )

    client = drms.Client()

    chunks = []

    current = START_DATE

    chunk_number = 1

    while current < END_DATE:

        chunk_end = min(
            current + timedelta(
                days=CHUNK_DAYS
            ),
            END_DATE,
        )

        print()
        print(
            f"========== CHUNK {chunk_number} =========="
        )

        result = query_chunk(
            client,
            current,
            chunk_end,
        )

        if not result.empty:
            chunks.append(result)

        current = chunk_end

        chunk_number += 1

    # ========================================================
    # COMBINE
    # ========================================================

    print()
    print("========================================")
    print(" COMBINING CHUNKS")
    print("========================================")

    if not chunks:
        raise RuntimeError(
            "No SHARP data were returned."
        )

    data = pd.concat(
        chunks,
        ignore_index=True,
    )

    rows_before = len(data)

    # Remove true duplicate SHARP observations.
    data = data.drop_duplicates(
        subset=[
            "HARPNUM",
            "T_REC",
        ]
    )

    duplicates_removed = (
        rows_before - len(data)
    )

    print(
        f"Duplicate observations removed: "
        f"{duplicates_removed}"
    )

    # Sort chronologically.
    data = data.sort_values(
        [
            "T_REC",
            "HARPNUM",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("========================================")
    print(" COLLECTION SUMMARY")
    print("========================================")

    print(
        f"Total rows: {len(data)}"
    )

    print(
        f"Unique HARPNUM: "
        f"{data['HARPNUM'].nunique()}"
    )

    print(
        f"Unique NOAA AR: "
        f"{data['NOAA_AR'].nunique()}"
    )

    print()
    print("Time range:")

    print(
        f"{data['T_REC'].min()} "
        f"→ "
        f"{data['T_REC'].max()}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========================================")
    print(" COLLECTION COMPLETE")
    print("========================================")

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()