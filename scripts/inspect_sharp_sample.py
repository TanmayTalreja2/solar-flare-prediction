from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/sharp/sharp_sample_2012_03_07.parquet"
)


def main() -> None:
    """Inspect the downloaded SHARP sample."""

    data = pd.read_parquet(DATA_PATH)

    print("========== DATASET OVERVIEW ==========")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print()

    print("========== DATA TYPES ==========")
    print(data.dtypes)
    print()

    print("========== MISSING VALUES ==========")
    print(data.isna().sum())
    print()

    print("========== ZERO VALUES ==========")

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns

    print(
        (data[numeric_columns] == 0).sum()
    )

    print()
    print("========== POTENTIALLY INVALID ROWS ==========")

    problem_mask = (
    (data["USFLUX"] == 0)
    & (data["TOTUSJH"] == 0)
    & (data["TOTPOT"] == 0)
    & (data["MEANPOT"].isna())
    & (data["MEANSHR"].isna())
)

    problem_rows = data.loc[
    problem_mask,
    [
        "T_REC",
        "HARPNUM",
        "NOAA_AR",
        "USFLUX",
        "TOTUSJH",
        "TOTPOT",
        "MEANPOT",
        "MEANSHR",
    ],
]

    print(f"Problematic rows: {len(problem_rows)}")
    print(problem_rows.head(20))

    print("========== ACTIVE REGIONS ==========")

    print(
        data["HARPNUM"]
        .nunique()
    )

    print()

    print("========== NOAA ACTIVE REGIONS ==========")

    print(
        data["NOAA_AR"]
        .dropna()
        .nunique()
    )

    print()

    print("========== SAMPLE ==========")
    print(data.head())


    print("========== CMASK ZERO OBSERVATIONS ==========")

    cmask_zero = data[data["CMASK"] == 0]

    print(f"Rows with CMASK = 0: {len(cmask_zero)}")
    print()

    print(
    cmask_zero[
        [
            "T_REC",
            "HARPNUM",
            "NOAA_AR",
            "QUALITY",
            "CMASK",
            "USFLUX",
            "TOTUSJH",
            "TOTPOT",
            "MEANPOT",
            "MEANSHR",
        ]
    ].head(50)
)


if __name__ == "__main__":
    main()