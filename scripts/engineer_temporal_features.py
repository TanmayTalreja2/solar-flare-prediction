from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

project_root = Path(__file__).parent.parent

FEATURE_DIR = project_root / "data" / "processed" / "features"

# Years we want to process
YEARS = list(range(2010, 2018))

BASE_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]

TIME_WINDOWS = {
    "1h": pd.Timedelta(hours=1),
    "3h": pd.Timedelta(hours=3),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
}



def load_data(year):
    """Load the yearly SHARP + GOES feature dataset."""

    input_path = (
        FEATURE_DIR /
        f"sharp_goes_features_{year}_full.parquet"
    )

    if not input_path.exists():
        print(f"[SKIP] Dataset not found: {input_path}")
        return None

    print("=" * 60)
    print(f" LOADING {year} DATASET")
    print("=" * 60)

    data = pd.read_parquet(input_path)

    print(f"Rows loaded: {len(data)}")
    print(f"Columns loaded: {len(data.columns)}")

    return data


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare observations chronologically within each active region."""

    print()
    print("========== PREPARING DATA ==========")

    data = data.copy()

    data["observation_time"] = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    # Remove invalid timestamps
    before_invalid = len(data)

    data = data.dropna(
        subset=["NOAA_AR", "observation_time"]
    )

    invalid_removed = before_invalid - len(data)

    # Remove duplicate NOAA_AR + timestamp observations
    before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=[
            "NOAA_AR",
            "observation_time",
        ],
        keep="first",
    )

    duplicates_removed = (
        before_duplicates - len(data)
    )

    print(
        f"Invalid timestamp/region rows removed: "
        f"{invalid_removed}"
    )

    print(
        f"Duplicate observations removed: "
        f"{duplicates_removed}"
    )

    # Sort chronologically within each active region
    data = data.sort_values(
        [
            "NOAA_AR",
            "observation_time",
        ]
    ).reset_index(drop=True)

    print(
        "Sorted by NOAA_AR and observation_time."
    )

    return data




def create_time_aware_lag(
    data: pd.DataFrame,
    feature: str,
    hours: int,
) -> pd.Series:
    """
    Get the most recent observation at or before
    the requested historical time.

    Reject the lag if the selected observation
    is too far from the requested timestamp.
    """

    result = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )

    target_delta = pd.Timedelta(hours=hours)

    # Maximum acceptable distance from target timestamp
    max_gap = pd.Timedelta(minutes=24)

    for ar, group in data.groupby(
        "NOAA_AR",
        sort=False,
    ):

        group = group.sort_values(
            "observation_time"
        )

        times = group[
            "observation_time"
        ]

        values = group[
            feature
        ]

        target_times = (
            times - target_delta
        )

        positions = times.searchsorted(
            target_times,
            side="right",
        ) - 1

        valid = positions >= 0

        lagged_values = np.full(
            len(group),
            np.nan,
            dtype=float,
        )

        if valid.any():

            valid_positions = positions[valid]

            historical_times = (
                times.iloc[
                    valid_positions
                ].to_numpy()
            )

            requested_times = (
                target_times.iloc[
                    np.where(valid)[0]
                ].to_numpy()
            )

            gaps = (
                requested_times
                - historical_times
            )

            acceptable = (
                gaps <= max_gap
            )

            valid_indices = (
                np.where(valid)[0]
            )

            accepted_indices = (
                valid_indices[acceptable]
            )

            accepted_positions = (
                valid_positions[acceptable]
            )

            lagged_values[
                accepted_indices
            ] = values.iloc[
                accepted_positions
            ].to_numpy()

        result.loc[
            group.index
        ] = lagged_values

    return result

def create_change_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("========== TIME-AWARE CHANGE FEATURES ==========")

    data = data.copy()

    for feature in BASE_FEATURES:

        for period, hours in {
            "1h": 1,
            "3h": 3,
            "6h": 6,
            "12h": 12,
        }.items():

            lagged = create_time_aware_lag(
                data,
                feature,
                hours,
            )

            new_feature = (
                f"{feature}_CHANGE_{period}"
            )

            data[new_feature] = (
                data[feature] - lagged
            )

            print(
                f"Created: {new_feature}"
            )

    return data


def create_relative_change_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print(
        "========== TIME-AWARE RELATIVE CHANGE FEATURES =========="
    )

    data = data.copy()

    for feature in BASE_FEATURES:

        for period, hours in {
            "1h": 1,
            "3h": 3,
            "6h": 6,
            "12h": 12,
        }.items():

            lagged = create_time_aware_lag(
                data,
                feature,
                hours,
            )

            new_feature = (
                f"{feature}_RELCHANGE_{period}"
            )

            denominator = lagged.abs()

            data[new_feature] = np.where(
                denominator > 0,
                (
                    data[feature] - lagged
                ) / denominator,
                np.nan,
            )

            print(
                f"Created: {new_feature}"
            )

    return data


def create_rolling_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create past-only time-based rolling standard deviations."""

    print()
    print("========== ROLLING FEATURES ==========")

    data = data.copy()

    for feature in BASE_FEATURES:

        for period, delta in TIME_WINDOWS.items():

            new_feature = (
                f"{feature}_ROLLSTD_{period}"
            )

            result = pd.Series(
                np.nan,
                index=data.index,
                dtype=float,
            )

            for region, group in data.groupby(
                "NOAA_AR",
                sort=False,
            ):

                group = group.sort_values(
                    "observation_time"
                )

                series = (
                    group[
                        [
                            "observation_time",
                            feature,
                        ]
                    ]
                    .set_index(
                        "observation_time"
                    )[feature]
                )

                rolling = (
                    series
                    .shift(1)
                    .rolling(
                        delta,
                        min_periods=3,
                    )
                    .std()
                )

                result.loc[
                    group.index
                ] = rolling.to_numpy()

            data[new_feature] = result

            print(
                f"Created: {new_feature}"
            )

    return data


def validate_features(
    data: pd.DataFrame,
) -> None:
    """Validate engineered temporal features."""

    print()
    print(
        "========== TEMPORAL FEATURE VALIDATION =========="
    )

    temporal_features = [
        column
        for column in data.columns
        if (
            "_CHANGE_" in column
            or "_RELCHANGE_" in column
            or "_ROLLSTD_" in column
        )
    ]

    print(
        f"Temporal features created: "
        f"{len(temporal_features)}"
    )

    for feature in temporal_features:

        infinite = np.isinf(
            data[feature]
        ).sum()

        missing = data[feature].isna().sum()

        print(
            f"{feature:30s} "
            f"infinite={infinite} "
            f"missing={missing}"
        )


def save_data(data, year):
    """Save the yearly temporal feature dataset."""

    output_path = (
        FEATURE_DIR /
        f"sharp_goes_temporal_features_{year}_full.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print(f" DATASET {year} SAVED")
    print("=" * 60)
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Output: {output_path}")


def main():

    print("=" * 70)
    print(" MULTI-YEAR TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    successful = []
    skipped = []

    for year in YEARS:

        data = load_data(year)

        if data is None:
            skipped.append(year)
            continue

        data = prepare_data(data)

        data = create_change_features(data)

        data = create_relative_change_features(data)

        data = create_rolling_features(data)

        validate_features(data)

        save_data(data, year)

        successful.append(year)

        print()
        print(f"✓ Finished {year}")
        print()

    print("=" * 70)
    print(" MULTI-YEAR FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(f"Successfully processed: {successful}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()


