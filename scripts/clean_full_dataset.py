from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_full_25params.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/cleaned/"
    "sharp_goes_clean_2012_full_25params.parquet"
)


PHYSICAL_COLUMNS = [
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


def main() -> None:

    print("========================================")
    print(" Q1 2012 DATA CLEANING")
    print("========================================")

    data = pd.read_parquet(
        INPUT_PATH
    ).copy()

    original_rows = len(data)

    print(
        f"Original rows: {original_rows}"
    )

    # --------------------------------------------------
    # Force physical SHARP columns to numeric.
    #
    # JSOC/DRMS occasionally returns a string error
    # sentinel (e.g. "Invalid KeyLink") instead of a
    # numeric value or NaN for some of the more exotic
    # derived parameters. Coerce these to NaN here so
    # downstream numeric operations never crash on a
    # stray string.
    # --------------------------------------------------

    print()
    print(
        "========== NUMERIC COERCION =========="
    )

    coerced_any = False

    for column in PHYSICAL_COLUMNS:

        if column not in data.columns:
            continue

        before_missing = data[column].isna().sum()

        coerced = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        new_invalid = (
            coerced.isna().sum()
            - before_missing
        )

        data[column] = coerced

        if new_invalid > 0:

            coerced_any = True

            print(
                f"{column:20s} "
                f"{new_invalid} invalid string values -> NaN"
            )

    if not coerced_any:

        print(
            "No invalid string values found "
            "(all columns already clean or numeric)."
        )
    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    invalid_time = (
        data["observation_time"].isna()
    ).sum()

    data = data[
        data["observation_time"].notna()
    ].copy()

    print()
    print(
        f"Invalid timestamps removed: "
        f"{invalid_time}"
    )

    # --------------------------------------------------
    # Active regions
    # --------------------------------------------------

    data["NOAA_AR"] = pd.to_numeric(
        data["NOAA_AR"],
        errors="coerce",
    )

    invalid_ar = (
        data["NOAA_AR"].isna()
        | (data["NOAA_AR"] == 0)
    ).sum()

    data = data[
        data["NOAA_AR"].notna()
        & (data["NOAA_AR"] != 0)
    ].copy()

    print(
        f"Invalid active regions removed: "
        f"{invalid_ar}"
    )

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    invalid_target = (
        data["target_24h"].isna()
        | ~data["target_24h"].isin([0, 1])
    ).sum()

    data = data[
        data["target_24h"].notna()
        & data["target_24h"].isin([0, 1])
    ].copy()

    print(
        f"Invalid targets removed: "
        f"{invalid_target}"
    )

    # --------------------------------------------------
    # True duplicate SHARP observations
    # --------------------------------------------------

    before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=[
            "HARPNUM",
            "T_REC",
        ]
    ).copy()

    duplicates_removed = (
        before_duplicates - len(data)
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )

    # --------------------------------------------------
    # Infinite values
    # --------------------------------------------------

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    infinite_mask = np.isinf(
        data[numeric_columns]
    ).any(axis=1)

    infinite_rows = (
        infinite_mask.sum()
    )

    data = data[
        ~infinite_mask
    ].copy()

    print(
        f"Infinite-value rows removed: "
        f"{infinite_rows}"
    )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    data = data.sort_values(
        [
            "observation_time",
            "HARPNUM",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        "========== CLEAN DATASET =========="
    )

    print(
        f"Final rows: {len(data)}"
    )

    print(
        f"Rows removed: "
        f"{original_rows - len(data)}"
    )

    print(
        f"Unique active regions: "
        f"{data['NOAA_AR'].nunique()}"
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        data["target_24h"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Missing values:"
    )

    missing = (
        data.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[missing > 0]
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Dataset saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "========== CLEANING COMPLETE =========="
    )


if __name__ == "__main__":
    main()