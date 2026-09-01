from pathlib import Path

import pandas as pd
import numpy as np


INPUT_PATH = Path(
    "results/temporal_model/test_predictions.csv"
)


def main():

    print("========================================")
    print(" PROBABILITY DISTRIBUTION ANALYSIS")
    print("========================================")

    data = pd.read_csv(INPUT_PATH)

    probability = data[
        "flare_probability"
    ]

    target = data[
        "target_24h"
    ]

    print()
    print("========== BASIC STATISTICS ==========")

    print(
        probability.describe()
    )

    print()
    print("========== PROBABILITY PERCENTILES ==========")

    percentiles = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
        100,
    ]

    for p in percentiles:

        value = np.percentile(
            probability,
            p,
        )

        print(
            f"{p:5.1f}% : {value:.6f}"
        )

    print()
    print("========== ACTUAL CLASS PROBABILITIES ==========")

    positive = probability[
        target == 1
    ]

    negative = probability[
        target == 0
    ]

    print(
        f"Positive samples: {len(positive)}"
    )

    print(
        f"Positive median probability: "
        f"{positive.median():.6f}"
    )

    print(
        f"Positive mean probability: "
        f"{positive.mean():.6f}"
    )

    print()

    print(
        f"Negative samples: {len(negative)}"
    )

    print(
        f"Negative median probability: "
        f"{negative.median():.6f}"
    )

    print(
        f"Negative mean probability: "
        f"{negative.mean():.6f}"
    )

    print()
    print("========== PROBABILITY BINS ==========")

    bins = [
        0,
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
        1.0,
    ]

    labels = [
        "0-0.001",
        "0.001-0.005",
        "0.005-0.01",
        "0.01-0.02",
        "0.02-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.50",
        "0.50-1.00",
    ]

    data["probability_bin"] = pd.cut(
        probability,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distribution = (
        data
        .groupby(
            "probability_bin",
            observed=False,
        )
        .agg(
            total=("target_24h", "count"),
            flares=("target_24h", "sum"),
        )
    )

    distribution["flare_rate"] = (
        distribution["flares"]
        / distribution["total"]
    )

    print(
        distribution
    )

    print()
    print("========================================")
    print(" ANALYSIS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()