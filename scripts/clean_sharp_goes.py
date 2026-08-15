from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_03_07.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/cleaned/"
    "sharp_goes_clean_2012_03_07.parquet"
)

FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]


def load_data() -> pd.DataFrame:
    """Load the aligned dataset."""

    print("Loading aligned dataset...")

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(data)}")

    return data


def validate_timestamps(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove observations with invalid timestamps."""

    print()
    print("========== TIMESTAMP VALIDATION ==========")

    before = len(data)

    data = data.dropna(
        subset=["observation_time"]
    ).copy()

    removed = before - len(data)

    print(
        f"Invalid timestamps removed: {removed}"
    )

    return data


def remove_invalid_regions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove observations without valid NOAA active regions."""

    print()
    print("========== ACTIVE REGION VALIDATION ==========")

    before = len(data)

    # NOAA_AR == 0 means there is no valid
    # NOAA active-region association.
    data = data[
        data["NOAA_AR"].notna()
        & (data["NOAA_AR"] > 0)
    ].copy()

    removed = before - len(data)

    print(
        f"Invalid active-region rows removed: {removed}"
    )

    return data


def remove_duplicates(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicate active-region observations."""

    print()
    print("========== DUPLICATE CHECK ==========")

    before = len(data)

    data = data.drop_duplicates(
        subset=[
            "NOAA_AR",
            "observation_time",
        ]
    ).copy()

    removed = before - len(data)

    print(
        f"Duplicate rows removed: {removed}"
    )

    return data


def handle_infinite_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Convert infinite feature values to NaN."""

    print()
    print("========== INFINITE VALUE CHECK ==========")

    infinite_count = np.isinf(
        data[FEATURES]
        .select_dtypes(
            include="number"
        )
    ).sum().sum()

    print(
        f"Infinite values found: {infinite_count}"
    )

    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


def validate_targets(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the binary prediction target."""

    print()
    print("========== TARGET VALIDATION ==========")

    valid_targets = {0, 1}

    invalid = ~data["target_24h"].isin(
        valid_targets
    )

    print(
        f"Invalid target rows: {invalid.sum()}"
    )

    data = data[
        ~invalid
    ].copy()

    return data


def print_feature_quality(
    data: pd.DataFrame,
) -> None:
    """Display feature quality statistics."""

    print()
    print("========== FEATURE QUALITY ==========")

    for feature in FEATURES:

        missing = data[feature].isna().sum()

        percentage = (
            missing / len(data)
        ) * 100

        print(
            f"{feature:10s} "
            f"missing={missing:4d} "
            f"({percentage:.2f}%)"
        )


def save_data(
    data: pd.DataFrame,
) -> None:
    """Save cleaned dataset."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========== DATASET SAVED ==========")
    print(
        f"Rows: {len(data)}"
    )
    print(
        f"Columns: {len(data.columns)}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )


def main() -> None:
    """Run the complete cleaning pipeline."""

    data = load_data()

    data = validate_timestamps(
        data
    )

    data = remove_invalid_regions(
        data
    )

    data = remove_duplicates(
        data
    )

    data = handle_infinite_values(
        data
    )

    data = validate_targets(
        data
    )

    print_feature_quality(
        data
    )

    save_data(
        data
    )

    print()
    print("========== CLEANING COMPLETE ==========")


if __name__ == "__main__":
    main()