from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/cleaned/"
    "sharp_goes_clean_2012_03_07.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_03_07.parquet"
)

BASE_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]


def load_data() -> pd.DataFrame:
    """Load the cleaned dataset."""

    print("Loading cleaned dataset...")

    data = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Rows loaded: {len(data)}"
    )

    return data


def create_log_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create log-transformed magnetic features."""

    print()
    print(
        "========== LOG FEATURES =========="
    )

    data = data.copy()

    log_features = [
        "USFLUX",
        "TOTUSJH",
        "TOTPOT",
        "MEANPOT",
    ]

    for feature in log_features:

        new_feature = f"LOG_{feature}"

        data[new_feature] = np.log1p(
            data[feature]
        )

        print(
            f"Created: {new_feature}"
        )

    return data


def create_temporal_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar-based temporal features."""

    print()
    print(
        "========== TEMPORAL FEATURES =========="
    )

    data = data.copy()

    timestamp = pd.to_datetime(
        data["observation_time"],
        errors="coerce",
    )

    data["observation_hour"] = (
        timestamp.dt.hour
    )

    data["day_of_year"] = (
        timestamp.dt.dayofyear
    )

    print(
        "Created: observation_hour"
    )

    print(
        "Created: day_of_year"
    )

    return data


def validate_features(
    data: pd.DataFrame,
) -> None:
    """Validate engineered features."""

    print()
    print(
        "========== FEATURE VALIDATION =========="
    )

    engineered = [
        column
        for column in data.columns
        if column.startswith("LOG_")
        or column in [
            "observation_hour",
            "day_of_year",
        ]
    ]

    for feature in engineered:

        infinite = np.isinf(
            data[feature]
        ).sum()

        print(
            f"{feature:20s} "
            f"infinite={infinite}"
        )


def save_data(
    data: pd.DataFrame,
) -> None:
    """Save the engineered dataset."""

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
        "========== DATASET SAVED =========="
    )

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
    """Run the feature-engineering pipeline."""

    data = load_data()

    data = create_log_features(
        data
    )

    data = create_temporal_features(
        data
    )

    validate_features(
        data
    )

    save_data(
        data
    )

    print()
    print(
        "========== FEATURE ENGINEERING COMPLETE =========="
    )


if __name__ == "__main__":
    main()