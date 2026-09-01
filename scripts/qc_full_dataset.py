from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_full.parquet"
)


def main() -> None:

    print("========================================")
    print(" FULL 2012 DATASET QUALITY CONTROL")
    print("========================================")

    data = pd.read_parquet(
        INPUT_PATH
    )

    print()
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    # --------------------------------------------------
    # Timestamp checks
    # --------------------------------------------------

    print()
    print("========== TIMESTAMP CHECK ==========")

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    print(
        f"Invalid timestamps: "
        f"{data['observation_time'].isna().sum()}"
    )

    print(
        f"First observation: "
        f"{data['observation_time'].min()}"
    )

    print(
        f"Last observation: "
        f"{data['observation_time'].max()}"
    )

    # --------------------------------------------------
    # Active region checks
    # --------------------------------------------------

    print()
    print("========== ACTIVE REGION CHECK ==========")

    print(
        f"Unique NOAA_AR: "
        f"{data['NOAA_AR'].nunique()}"
    )

    print(
        f"NOAA_AR = 0: "
        f"{(data['NOAA_AR'] == 0).sum()}"
    )

    print(
        f"Missing NOAA_AR: "
        f"{data['NOAA_AR'].isna().sum()}"
    )

    # --------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------

    print()
    print("========== DUPLICATE CHECK ==========")

    if "HARPNUM" in data.columns:

        duplicates = data.duplicated(
            subset=[
                "HARPNUM",
                "T_REC",
            ]
        ).sum()

    else:

        duplicates = data.duplicated(
            subset=[
                "NOAA_AR",
                "observation_time",
            ]
        ).sum()

    print(
        f"Duplicate SHARP observations: "
        f"{duplicates}"
    )

    # --------------------------------------------------
    # Target checks
    # --------------------------------------------------

    print()
    print("========== TARGET CHECK ==========")

    print(
        data["target_24h"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print()

    print(
        "Missing targets:",
        data["target_24h"].isna().sum(),
    )

    print(
        "Invalid targets:",
        (
            ~data["target_24h"].isin([0, 1])
        ).sum(),
    )

    # --------------------------------------------------
    # Infinite values
    # --------------------------------------------------

    print()
    print("========== NUMERIC CHECK ==========")

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    infinite_values = np.isinf(
        data[numeric_columns]
    ).sum().sum()

    print(
        f"Infinite numeric values: "
        f"{infinite_values}"
    )

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print()
    print("========== MISSING VALUES ==========")

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
    # Monthly target distribution
    # --------------------------------------------------

    print()
    print("========== MONTHLY TARGET DISTRIBUTION ==========")

    data["month"] = (
        data["observation_time"]
        .dt.to_period("M")
    )

    monthly = (
        data.groupby("month")[
            "target_24h"
        ]
        .agg(
            observations="count",
            positives="sum",
        )
    )

    monthly["positive_rate"] = (
        monthly["positives"]
        / monthly["observations"]
        * 100
    )

    print(
        monthly
    )

    # --------------------------------------------------
    # Positive observations by region
    # --------------------------------------------------

    print()
    print(
        "========== POSITIVE EVENTS BY REGION =========="
    )

    positive_by_region = (
        data[data["target_24h"] == 1]
        .groupby("NOAA_AR")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        positive_by_region
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print()
    print("========================================")
    print(" Q1 QC COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()