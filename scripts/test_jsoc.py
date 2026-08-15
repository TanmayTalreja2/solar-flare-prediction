import drms


def main() -> None:
    """Query a small SHARP sample from JSOC."""

    client = drms.Client()

    query = (
        "hmi.sharp_cea_720s"
        "[][2012.03.07_00:00:00_TAI/1d@12m]"
    )

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

    print(f"Rows returned: {len(result)}")
    print()

    print("Columns:")
    print(result.columns.tolist())
    print()

    print("First 5 rows:")
    print(result.head())

    print()
    print("========== QUALITY VALUES ==========")
    print(result["QUALITY"].value_counts(dropna=False))

    print()
    print("========== CMASK STATISTICS ==========")
    print(result["CMASK"].describe())

    print()
    print("========== MASK VALUES ==========")
    print(result["MASK"].value_counts(dropna=False))


if __name__ == "__main__":
    main()