from pathlib import Path

import drms


OUTPUT_PATH = Path("data/raw/sharp/sharp_sample_2012_03_07.parquet")


def main() -> None:
    """Download and save a small SHARP sample from JSOC."""

    client = drms.Client()

    query = (
        "hmi.sharp_cea_720s"
        "[][2012.03.07_00:00:00_TAI/1d@12m]"
    )

    print("Connecting to JSOC...")
    print(f"Query: {query}")

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Rows saved: {len(result)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()