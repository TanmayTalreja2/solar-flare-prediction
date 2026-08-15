import drms
from pathlib import Path


OUTPUT_PATH = Path(
    "data/raw/sharp/sharp_2012_03_01_7d.parquet"
)


def main() -> None:
    """Collect a 30-day SHARP dataset from JSOC."""

    client = drms.Client()

    query = (
        "hmi.sharp_cea_720s"
        "[][2012.03.01_00:00:00_TAI/7d@12m]"
    )

    print("========================================")
    print(" SHARP 7-DAY COLLECTION")
    print("========================================")
    print()

    print("Connecting to JSOC...")
    print(f"Query: {query}")
    print()

    result = client.query(
        query,
        key=[
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
        ],
        skip_conversion=["QUALITY"],
    )

    print()
    print("========== COLLECTION RESULTS ==========")
    print(f"Rows returned: {len(result)}")
    print()

    print("Columns:")
    print(result.columns.tolist())
    print()

    print("First 5 rows:")
    print(result.head())
    print()

    print("========== ACTIVE REGIONS ==========")

    print(
        f"Unique HARPNUM: "
        f"{result['HARPNUM'].nunique()}"
    )

    print(
        f"Unique NOAA_AR: "
        f"{result['NOAA_AR'].nunique()}"
    )

    print()

    print("========== TIME RANGE ==========")

    print(
        f"First observation: "
        f"{result['T_REC'].min()}"
    )

    print(
        f"Last observation: "
        f"{result['T_REC'].max()}"
    )

    print()

    print("========== QUALITY VALUES ==========")
    print(
        result["QUALITY"].value_counts(
            dropna=False
        )
    )

    print()

    print("========== CMASK STATISTICS ==========")
    print(result["CMASK"].describe())

    print()

    print("========== MASK VALUES ==========")
    print(
        result["MASK"].value_counts(
            dropna=False
        )
    )

    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========== DATASET SAVED ==========")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")


if __name__ == "__main__":
    main()