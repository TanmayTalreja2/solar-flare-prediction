from pathlib import Path
import pandas as pd


LABEL_PATH = Path(
    "data/processed/magnetograms/dataset_labels.csv"
)

MAGNETOGRAM_DIR = Path(
    "data/processed/magnetograms"
)


def build_filename(row):

    timestamp = pd.to_datetime(
        row["observation_time"]
    )

    time_string = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"harp_{int(row['HARPNUM'])}_"
        f"{time_string}_t0.npz"
    )


def main():

    print("=" * 60)
    print(" MAGNETOGRAM PATH DIAGNOSTIC")
    print("=" * 60)

    labels = pd.read_csv(
        LABEL_PATH
    )

    labels["observation_time"] = pd.to_datetime(
        labels["observation_time"]
    )

    print()
    print(f"Total labels: {len(labels)}")

    labels["filename"] = labels.apply(
        build_filename,
        axis=1,
    )

    labels["file_path"] = labels["filename"].apply(
        lambda name: MAGNETOGRAM_DIR / name
    )

    labels["file_exists"] = labels[
        "file_path"
    ].apply(
        lambda path: path.exists()
    )

    print()
    print("=" * 60)
    print(" OVERALL FILE MATCHING")
    print("=" * 60)

    print(
        f"Files found: "
        f"{labels['file_exists'].sum()}"
    )

    print(
        f"Files missing: "
        f"{(~labels['file_exists']).sum()}"
    )

    print()
    print("=" * 60)
    print(" MATCHING BY CLASS")
    print("=" * 60)

    summary = (
        labels.groupby("target_24h")[
            "file_exists"
        ]
        .agg(
            total="count",
            found="sum",
        )
    )

    summary["missing"] = (
        summary["total"]
        - summary["found"]
    )

    print(summary)

    print()
    print("=" * 60)
    print(" SAMPLE POSITIVE ROWS")
    print("=" * 60)

    positives = labels[
        labels["target_24h"] == 1
    ]

    print(
        positives[
            [
                "HARPNUM",
                "observation_time",
                "filename",
                "file_exists",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 60)
    print(" SAMPLE NEGATIVE ROWS")
    print("=" * 60)

    negatives = labels[
        labels["target_24h"] == 0
    ]

    print(
        negatives[
            [
                "HARPNUM",
                "observation_time",
                "filename",
                "file_exists",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 60)
    print(" MISSING POSITIVE EXAMPLES")
    print("=" * 60)

    missing_positive = labels[
        (labels["target_24h"] == 1)
        &
        (~labels["file_exists"])
    ]

    print(
        missing_positive[
            [
                "HARPNUM",
                "observation_time",
                "filename",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()