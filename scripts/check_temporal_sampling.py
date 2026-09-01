from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_full.parquet"
)


def load_data() -> pd.DataFrame:
    """Load the feature dataset."""

    print("========================================")
    print(" TEMPORAL SAMPLING DIAGNOSTIC")
    print("========================================")

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows: {len(data)}")

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    return data


def check_duplicate_timestamps(
    data: pd.DataFrame,
) -> None:
    """Check duplicate NOAA_AR + timestamp combinations."""

    print()
    print("========== DUPLICATE TIMESTAMPS ==========")

    duplicates = data.duplicated(
        subset=["NOAA_AR", "observation_time"],
        keep=False,
    )

    duplicate_rows = data[duplicates]

    duplicate_pairs = (
        duplicate_rows[
            ["NOAA_AR", "observation_time"]
        ]
        .drop_duplicates()
    )

    print(
        f"Duplicate rows: {len(duplicate_rows)}"
    )

    print(
        f"Duplicate timestamp pairs: "
        f"{len(duplicate_pairs)}"
    )

    if len(duplicate_pairs) > 0:

        print()
        print("Sample duplicate pairs:")

        print(
            duplicate_pairs
            .head(10)
            .to_string(index=False)
        )


def check_identical_duplicates(
    data: pd.DataFrame,
) -> None:
    """Check whether duplicate observations are identical."""

    print()
    print("========== DUPLICATE CONTENT CHECK ==========")

    duplicated = data[
        data.duplicated(
            subset=[
                "NOAA_AR",
                "observation_time",
            ],
            keep=False,
        )
    ].copy()

    if duplicated.empty:

        print("No duplicate observations found.")

        return

    feature_columns = [
        "USFLUX",
        "TOTUSJH",
        "TOTPOT",
        "MEANPOT",
        "MEANSHR",
    ]

    identical_groups = 0
    non_identical_groups = 0

    grouped = duplicated.groupby(
        [
            "NOAA_AR",
            "observation_time",
        ]
    )

    for _, group in grouped:

        if group[feature_columns].nunique().max() <= 1:
            identical_groups += 1
        else:
            non_identical_groups += 1

    print(
        f"Identical duplicate groups: "
        f"{identical_groups}"
    )

    print(
        f"Non-identical duplicate groups: "
        f"{non_identical_groups}"
    )


def calculate_sampling_intervals(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate sampling intervals within each active region."""

    print()
    print("========== SAMPLING INTERVALS ==========")

    data = data.sort_values(
        [
            "NOAA_AR",
            "observation_time",
        ]
    ).copy()

    data["time_diff_minutes"] = (
        data
        .groupby("NOAA_AR")["observation_time"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    intervals = (
        data["time_diff_minutes"]
        .dropna()
        .value_counts()
        .sort_index()
    )

    print()

    print(
        intervals
        .head(20)
        .to_string()
    )

    print()

    print(
        f"Total intervals: "
        f"{len(data['time_diff_minutes'].dropna())}"
    )

    print(
        f"Normal 12-minute intervals: "
        f"{(data['time_diff_minutes'] == 12).sum()}"
    )

    print(
        f"Duplicate 0-minute intervals: "
        f"{(data['time_diff_minutes'] == 0).sum()}"
    )

    print(
        f"Intervals > 12 minutes: "
        f"{(data['time_diff_minutes'] > 12).sum()}"
    )

    return data


def check_large_gaps(
    data: pd.DataFrame,
) -> None:
    """Check for unusually large sampling gaps."""

    print()
    print("========== LARGE GAPS ==========")

    intervals = data["time_diff_minutes"].dropna()

    thresholds = [
        24,
        60,
        360,
        720,
    ]

    for threshold in thresholds:

        count = (
            intervals > threshold
        ).sum()

        print(
            f"> {threshold} minutes: "
            f"{count}"
        )

    large_gaps = data[
        data["time_diff_minutes"] > 60
    ]

    if not large_gaps.empty:

        print()
        print("Sample large gaps:")

        print(
            large_gaps[
                [
                    "NOAA_AR",
                    "observation_time",
                    "time_diff_minutes",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


def print_summary(
    data: pd.DataFrame,
) -> None:
    """Print final diagnostic summary."""

    print()
    print("========================================")
    print(" SAMPLING SUMMARY")
    print("========================================")

    intervals = data["time_diff_minutes"].dropna()

    print(
        f"Minimum interval: "
        f"{intervals.min():.1f} minutes"
    )

    print(
        f"Median interval: "
        f"{intervals.median():.1f} minutes"
    )

    print(
        f"Maximum interval: "
        f"{intervals.max():.1f} minutes"
    )

    normal_percentage = (
        (intervals == 12).mean() * 100
    )

    print(
        f"12-minute intervals: "
        f"{normal_percentage:.2f}%"
    )

    print()
    print("========== DIAGNOSTIC COMPLETE ==========")


def main() -> None:

    data = load_data()

    check_duplicate_timestamps(
        data
    )

    check_identical_duplicates(
        data
    )

    data = calculate_sampling_intervals(
        data
    )

    check_large_gaps(
        data
    )

    print_summary(
        data
    )


if __name__ == "__main__":
    main()