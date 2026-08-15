from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_03.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/cleaned/"
    "sharp_goes_clean_2012_03.parquet"
)


def main() -> None:
    """Clean the 30-day aligned dataset."""

    print("========================================")
    print(" 30-DAY DATA CLEANING")
    print("========================================")

    data = pd.read_parquet(
        INPUT_PATH
    )

    original_rows = len(data)

    print()
    print(
        f"Original rows: {original_rows}"
    )

    # --------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    invalid_timestamps = (
        data["observation_time"].isna()
    ).sum()

    data = data.dropna(
        subset=["observation_time"]
    )

    print(
        f"Invalid timestamps removed: "
        f"{invalid_timestamps}"
    )

    # --------------------------------------------------
    # Active region validation
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
    # Target validation
    # --------------------------------------------------

    data = data[
        data["target_24h"].isin([0, 1])
    ].copy()

    print(
        f"Invalid targets removed: "
        f"{original_rows - len(data)}"
    )

    # --------------------------------------------------
    # Remove duplicate SHARP observations
    # --------------------------------------------------

    before_duplicates = len(data)

    if "HARPNUM" in data.columns:
        data = data.drop_duplicates(
            subset=[
                "HARPNUM",
                "T_REC",
            ]
        )
    else:
        data = data.drop_duplicates(
            subset=[
                "NOAA_AR",
                "observation_time",
            ]
        )

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

    infinite_count = (
        infinite_mask.sum()
    )

    data = data[
        ~infinite_mask
    ].copy()

    print(
        f"Infinite-value rows removed: "
        f"{infinite_count}"
    )

    # --------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------

    data = data.sort_values(
        [
            "observation_time",
            "NOAA_AR",
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