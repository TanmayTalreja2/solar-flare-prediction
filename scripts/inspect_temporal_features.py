from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_temporal_features_2012_full.parquet"
)


def main():

    data = pd.read_parquet(INPUT_PATH)

    data["observation_time"] = pd.to_datetime(
        data["observation_time"]
    )

    print("========================================")
    print(" TEMPORAL FEATURE INSPECTION")
    print("========================================")

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    # Pick a region with plenty of observations
    region_counts = (
        data["NOAA_AR"]
        .value_counts()
    )

    region = region_counts.index[0]

    print()
    print(f"Inspecting NOAA_AR: {region}")
    print(
        f"Observations: "
        f"{region_counts.iloc[0]}"
    )

    region_data = (
        data[
            data["NOAA_AR"] == region
        ]
        .sort_values("observation_time")
        .copy()
    )

    columns = [
        "NOAA_AR",
        "observation_time",
        "USFLUX",
        "USFLUX_CHANGE_1h",
        "USFLUX_CHANGE_3h",
        "USFLUX_CHANGE_6h",
        "USFLUX_CHANGE_12h",
        "TOTUSJH",
        "TOTUSJH_CHANGE_1h",
        "TOTUSJH_CHANGE_3h",
    ]

    print()
    print(region_data[columns].head(20).to_string())

    print()
    print("========== TIME DIFFERENCES ==========")

    region_data["time_diff"] = (
        region_data["observation_time"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    print(
        region_data[
            [
                "observation_time",
                "time_diff",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print("========== SAMPLING SUMMARY ==========")

    print(
        region_data["time_diff"]
        .value_counts()
        .sort_index()
        .head(20)
    )


if __name__ == "__main__":
    main()