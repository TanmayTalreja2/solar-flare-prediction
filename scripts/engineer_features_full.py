from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/cleaned/"
    "sharp_goes_clean_2012_full_25params.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_full_25params.parquet"
)


BASE_FEATURES = [
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

LOG_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
]


def load_data() -> pd.DataFrame:
    print("Loading cleaned dataset...")
    data = pd.read_parquet(INPUT_PATH)
    print(f"Rows loaded: {len(data)}")
    print(f"Columns loaded: {len(data.columns)}")
    return data


def create_log_features(data: pd.DataFrame) -> pd.DataFrame:
    print()
    print("========== LOG FEATURES ==========")
    data = data.copy()
    for feature in LOG_FEATURES:
        new_feature = f"LOG_{feature}"
        data[new_feature] = np.log1p(data[feature])
        print(f"Created: {new_feature}")
    return data


def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    print()
    print("========== TEMPORAL FEATURES ==========")
    data = data.copy()
    timestamp = pd.to_datetime(data["observation_time"], errors="coerce")
    data["observation_hour"] = timestamp.dt.hour
    data["day_of_year"] = timestamp.dt.dayofyear
    print("Created: observation_hour")
    print("Created: day_of_year")
    return data


def validate_features(data: pd.DataFrame) -> None:
    print()
    print("========== FEATURE VALIDATION ==========")
    engineered = [
        column for column in data.columns
        if column.startswith("LOG_") or column in ["observation_hour", "day_of_year"]
    ]
    for feature in engineered:
        infinite = np.isinf(data[feature]).sum()
        missing = data[feature].isna().sum()
        print(f"{feature:20s} infinite={infinite} missing={missing}")

    print()
    print("========== BASE FEATURE MISSING VALUES ==========")
    for feature in BASE_FEATURES:
        if feature not in data.columns:
            print(f"{feature:20s} NOT FOUND IN DATA")
            continue
        missing = data[feature].isna().sum()
        print(f"{feature:20s} missing={missing}")


def save_data(data: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)
    print()
    print("========== DATASET SAVED ==========")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Output: {OUTPUT_PATH}")


def main() -> None:
    print("========================================")
    print(" FULL 2012 (25-PARAM) FEATURE ENGINEERING")
    print("========================================")
    data = load_data()
    data = create_log_features(data)
    data = create_temporal_features(data)
    validate_features(data)
    save_data(data)
    print()
    print("========== FEATURE ENGINEERING COMPLETE ==========")


if __name__ == "__main__":
    main()