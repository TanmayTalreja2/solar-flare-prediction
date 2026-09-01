from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/features/"
    "sharp_goes_features_2012_full.parquet"
)


def main():

    data = pd.read_parquet(
        INPUT_PATH
    ).copy()

    data["observation_time"] = pd.to_datetime(
        data["observation_time"]
    )

    data = data.sort_values(
        "observation_time"
    )

    print("========================================")
    print(" FULL 2012 TEMPORAL STRUCTURE")
    print("========================================")

    # --------------------------------------------------
    # Monthly distribution
    # --------------------------------------------------

    data["month"] = (
        data["observation_time"]
        .dt.to_period("M")
    )

    monthly = (
        data.groupby("month")["target_24h"]
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

    print()
    print("========== MONTHLY DISTRIBUTION ==========")
    print(monthly)

    # --------------------------------------------------
    # Active-region lifetime
    # --------------------------------------------------

    regions = (
        data.groupby("NOAA_AR")[
            "observation_time"
        ]
        .agg(
            first_observation="min",
            last_observation="max",
            observations="count",
        )
    )

    regions["lifetime_days"] = (
        regions["last_observation"]
        - regions["first_observation"]
    ).dt.total_seconds() / 86400

    print()
    print("========== ACTIVE REGION LIFETIMES ==========")

    print(
        regions["lifetime_days"]
        .describe()
    )

    # --------------------------------------------------
    # Positive regions
    # --------------------------------------------------

    positive_regions = (
        data[data["target_24h"] == 1]
        .groupby("NOAA_AR")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "========== POSITIVE-PRODUCING REGIONS =========="
    )

    print(
        f"Total: {len(positive_regions)}"
    )

    print()
    print(
        positive_regions.to_string()
    )

    # --------------------------------------------------
    # Region appearance by month
    # --------------------------------------------------

    first_seen = (
        regions["first_observation"]
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "========== NEW ACTIVE REGIONS BY MONTH =========="
    )

    print(
        first_seen
    )

    # --------------------------------------------------
    # Candidate temporal cutoffs
    # --------------------------------------------------

    print()
    print(
        "========== CANDIDATE CUTS =========="
    )

    for month in [
        "2012-04-01",
        "2012-05-01",
        "2012-06-01",
        "2012-07-01",
        "2012-08-01",
        "2012-09-01",
        "2012-10-01",
    ]:

        cutoff = pd.Timestamp(
            month
        )

        train = data[
            data["observation_time"] < cutoff
        ]

        test = data[
            data["observation_time"] >= cutoff
        ]

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

        test_positive_regions = set(
            test.loc[
                test["target_24h"] == 1,
                "NOAA_AR",
            ].unique()
        )

        train_positive_regions = set(
            train.loc[
                train["target_24h"] == 1,
                "NOAA_AR",
            ].unique()
        )

        shared_positive_regions = (
            test_positive_regions
            & train_positive_regions
        )

        print()
        print(
            f"Cutoff: {month}"
        )

        print(
            f"Train rows: {len(train):,}"
        )

        print(
            f"Test rows: {len(test):,}"
        )

        print(
            f"Train regions: {len(train_regions)}"
        )

        print(
            f"Test regions: {len(test_regions)}"
        )

        print(
            f"Overlapping regions: {len(overlap)}"
        )

        print(
            f"Train positive regions: "
            f"{len(train_positive_regions)}"
        )

        print(
            f"Test positive regions: "
            f"{len(test_positive_regions)}"
        )

        print(
            f"Shared positive regions: "
            f"{len(shared_positive_regions)}"
        )


if __name__ == "__main__":
    main()