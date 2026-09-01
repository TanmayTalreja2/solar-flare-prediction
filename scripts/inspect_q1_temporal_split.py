from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_q1.parquet"
)


def main():

    data = pd.read_parquet(
        INPUT_PATH
    ).copy()

    data["observation_time"] = pd.to_datetime(
        data["observation_time"]
    )

    cutoff = pd.Timestamp(
        "2012-03-01"
    )

    train = data[
        data["observation_time"] < cutoff
    ].copy()

    test = data[
        data["observation_time"] >= cutoff
    ].copy()

    train_regions = set(
        train["NOAA_AR"].unique()
    )

    test_regions = set(
        test["NOAA_AR"].unique()
    )

    overlap = (
        train_regions
        & test_regions
    )

    unseen_test = (
        test_regions
        - train_regions
    )

    print(
        "========================================"
    )
    print(
        " Q1 TEMPORAL SPLIT DIAGNOSTIC"
    )
    print(
        "========================================"
    )

    print()
    print(
        f"Training regions: {len(train_regions)}"
    )

    print(
        f"Testing regions:  {len(test_regions)}"
    )

    print(
        f"Overlapping regions: {len(overlap)}"
    )

    print(
        f"Unseen test regions: {len(unseen_test)}"
    )

    print()
    print(
        "Unseen test regions:"
    )

    print(
        sorted(unseen_test)
    )

    print()
    print(
        "========== TRAIN POSITIVES BY REGION =========="
    )

    train_positive = (
        train[
            train["target_24h"] == 1
        ]
        .groupby("NOAA_AR")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        train_positive
    )

    print()
    print(
        "========== TEST POSITIVES BY REGION =========="
    )

    test_positive = (
        test[
            test["target_24h"] == 1
        ]
        .groupby("NOAA_AR")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        test_positive
    )

    print()
    print(
        "========== REGION OVERLAP =========="
    )

    overlap_positive = (
        test_positive
        .index
        .intersection(
            train_positive.index
        )
    )

    print(
        "Test positive regions also positive in training:"
    )

    print(
        sorted(overlap_positive)
    )


if __name__ == "__main__":
    main()