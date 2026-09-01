from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)

BASE_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]

WINDOWS = ["1h", "3h", "6h", "12h"]


def main():

    print("========================================")
    print(" TEMPORAL FEATURE DIAGNOSTIC")
    print("========================================")

    data = pd.read_parquet(INPUT_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data = data.sort_values(
        ["NOAA_AR", "observation_time"]
    ).reset_index(drop=True)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    # --------------------------------------------------
    # BASIC CHECK
    # --------------------------------------------------

    print()
    print("========== BASIC CHECK ==========")

    print(
        f"Missing timestamps: "
        f"{data['observation_time'].isna().sum()}"
    )

    print(
        f"Unique active regions: "
        f"{data['NOAA_AR'].nunique()}"
    )

    print(
        f"Positive observations: "
        f"{data['target_24h'].sum()}"
    )

    # --------------------------------------------------
    # TEMPORAL FEATURE COVERAGE
    # --------------------------------------------------

    print()
    print("========== TEMPORAL FEATURE COVERAGE ==========")

    temporal_features = [
        c for c in data.columns
        if (
            "_CHANGE_" in c
            or "_RELCHANGE_" in c
            or "_ROLLSTD_" in c
        )
    ]

    for feature in temporal_features:

        valid = data[feature].notna().sum()
        missing = data[feature].isna().sum()

        print(
            f"{feature:35s} "
            f"valid={valid:7d} "
            f"missing={missing:7d} "
            f"coverage={valid / len(data):.3f}"
        )

    # --------------------------------------------------
    # CHECK CHANGE FEATURE TIMING
    # --------------------------------------------------

    print()
    print("========== CHANGE FEATURE TIMING ==========")

    grouped = data.groupby(
        "NOAA_AR",
        sort=False,
    )

    for feature in BASE_FEATURES:

        for window in WINDOWS:

            change_col = (
                f"{feature}_CHANGE_{window}"
            )

            if change_col not in data.columns:
                continue

            # Find previous valid observation
            previous_time = grouped[
                "observation_time"
            ].shift(1)

            time_diff = (
                data["observation_time"]
                - previous_time
            ).dt.total_seconds() / 3600

            valid = data[change_col].notna()

            if valid.sum() == 0:
                continue

            print(
                f"{change_col:35s} "
                f"current-prev median="
                f"{time_diff[valid].median():.2f}h"
            )

            break

    # --------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------

    print()
    print("========== DUPLICATE TIMESTAMP CHECK ==========")

    duplicates = data.duplicated(
        subset=[
        "NOAA_AR",
        "observation_time",
        ],
        keep=False,
)

    duplicate_groups = data.loc[
        duplicates,
        ["NOAA_AR", "observation_time"]
        ].drop_duplicates()

    print(
    f"Rows involved in duplicate timestamps: "
    f"{duplicates.sum()}"
)

    print(
    f"Duplicate timestamp groups: "
    f"{len(duplicate_groups)}"
)

    # --------------------------------------------------
    # TARGET + TEMPORAL COVERAGE
    # --------------------------------------------------

    print()
    print("========== POSITIVE OBSERVATION COVERAGE ==========")

    positives = data["target_24h"] == 1

    for window in WINDOWS:

        columns = [
            f"{feature}_CHANGE_{window}"
            for feature in BASE_FEATURES
        ]

        columns = [
            c for c in columns
            if c in data.columns
        ]

        valid_all = data[columns].notna().all(axis=1)

        positive_valid = (
            positives & valid_all
        ).sum()

        positive_total = positives.sum()

        print(
            f"{window:4s} "
            f"positive coverage: "
            f"{positive_valid}/{positive_total} "
            f"("
            f"{positive_valid / positive_total:.2%}"
            f")"
        )

    # --------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------

    print()
    print("========== LEAKAGE CHECK ==========")

    suspicious = []

    for column in temporal_features:

        if data[column].dtype.kind not in "biufc":
            continue

        correlation = data[
            [column, "target_24h"]
        ].corr().iloc[0, 1]

        if abs(correlation) > 0.8:
            suspicious.append(
                (column, correlation)
            )

    if suspicious:

        print(
            "Potentially suspicious correlations:"
        )

        for column, correlation in suspicious:
            print(
                f"{column:35s} "
                f"correlation={correlation:.4f}"
            )

    else:

        print(
            "No obviously suspicious "
            "correlations detected."
        )

    # --------------------------------------------------

    print()
    print("========================================")
    print(" TEMPORAL DIAGNOSTIC COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()