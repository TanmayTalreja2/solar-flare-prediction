from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_03_07.parquet"
)

FIGURE_DIR = Path(
    "reports/figures"
)

FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]


def load_data() -> pd.DataFrame:
    """Load the aligned SHARP-GOES dataset."""

    print("Loading aligned dataset...")

    data = pd.read_parquet(DATA_PATH)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


def print_dataset_overview(
    data: pd.DataFrame,
) -> None:
    """Print basic dataset information."""

    print()
    print("========== DATASET OVERVIEW ==========")

    print(data.info())

    print()
    print("========== TARGET DISTRIBUTION ==========")

    print(
        data["target_24h"]
        .value_counts()
    )

    print()
    print("Target percentages:")

    print(
        data["target_24h"]
        .value_counts(
            normalize=True
        ).mul(100).round(2)
    )


def analyze_missing_values(
    data: pd.DataFrame,
) -> None:
    """Analyze missing values."""

    print()
    print("========== MISSING VALUES ==========")

    missing = (
        data[FEATURES]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(missing)

    missing_percent = (
        data[FEATURES]
        .isna()
        .mean()
        .mul(100)
        .round(2)
    )

    print()
    print("Missing percentages:")
    print(missing_percent)


def plot_target_distribution(
    data: pd.DataFrame,
) -> None:
    """Create target distribution plot."""

    counts = (
        data["target_24h"]
        .value_counts()
        .sort_index()
    )

    labels = [
        "No M/X flare",
        "M/X flare",
    ]

    values = [
        counts.get(0, 0),
        counts.get(1, 0),
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values,
    )

    plt.title(
        "24-Hour M/X Flare Target Distribution"
    )

    plt.ylabel("Number of observations")

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "target_distribution.png"
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output}")


def plot_feature_distributions(
    data: pd.DataFrame,
) -> None:
    """Create individual feature distribution plots."""

    for feature in FEATURES:

        plt.figure(figsize=(8, 5))

        data[feature].dropna().plot(
            kind="hist",
            bins=40,
        )

        plt.title(
            f"Distribution of {feature}"
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        output = (
            FIGURE_DIR
            / f"{feature.lower()}_distribution.png"
        )

        plt.savefig(
            output,
            dpi=200,
        )

        plt.close()

        print(f"Saved: {output}")


def plot_feature_correlation(
    data: pd.DataFrame,
) -> None:
    """Create feature correlation heatmap."""

    correlation = (
        data[FEATURES]
        .corr()
    )

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        correlation,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Correlation"
    )

    plt.xticks(
        range(len(FEATURES)),
        FEATURES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(FEATURES)),
        FEATURES,
    )

    plt.title(
        "SHARP Feature Correlation"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "feature_correlation.png"
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output}")

    print()
    print("========== CORRELATION MATRIX ==========")
    print(correlation.round(3))


def compare_target_groups(
    data: pd.DataFrame,
) -> None:
    """Compare feature statistics between target groups."""

    print()
    print(
        "========== FEATURE STATISTICS BY TARGET =========="
    )

    comparison = (
        data.groupby("target_24h")[FEATURES]
        .median()
        .T
    )

    print(
        comparison
    )

    comparison.to_csv(
        FIGURE_DIR
        / "feature_target_medians.csv"
    )


def main() -> None:
    """Run the complete EDA pipeline."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    print_dataset_overview(
        data
    )

    analyze_missing_values(
        data
    )

    plot_target_distribution(
        data
    )

    plot_feature_distributions(
        data
    )

    plot_feature_correlation(
        data
    )

    compare_target_groups(
        data
    )

    print()
    print("========== EDA COMPLETE ==========")
    print(
        f"Figures saved to: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()