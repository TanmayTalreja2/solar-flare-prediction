import pandas as pd
from pathlib import Path
import re

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
    / "dataset_labels.csv"
)

MAGNETOGRAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"
    / "dataset_labels_with_paths.csv"
)

MAX_TIME_DIFFERENCE_MINUTES = 15


# ============================================================
# PARSE MAGNETOGRAM FILENAME
# ============================================================

def parse_magnetogram_filename(filename):

    pattern = (
        r"harp_(\d+)_"
        r"(\d{8})_"
        r"(\d{6})_"
        r"(t[01])\.npz"
    )

    match = re.match(pattern, filename)

    if match is None:
        return None

    harpnum = int(match.group(1))

    date_part = match.group(2)
    time_part = match.group(3)
    suffix = match.group(4)

    timestamp = pd.to_datetime(
        date_part + time_part,
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    if pd.isna(timestamp):
        return None

    return {
        "HARPNUM": harpnum,
        "file_time": timestamp,
        "file_path": filename,
        "file_suffix": suffix
    }
# ============================================================
# LOAD MAGNETOGRAM FILE INDEX
# ============================================================

def build_file_index():

    print()
    print("=" * 60)
    print("INDEXING MAGNETOGRAM FILES")
    print("=" * 60)

    records = []

    files = list(MAGNETOGRAM_DIR.glob("*.npz"))

    print(f"NPZ files found: {len(files)}")

    for file_path in files:

        parsed = parse_magnetogram_filename(
            file_path.name
        )

        if parsed is not None:
            records.append(parsed)

    index = pd.DataFrame(records)

    print(f"Successfully indexed: {len(index)}")

    if len(index) == 0:
        raise ValueError(
            "No magnetogram files could be parsed."
        )

    index = index.sort_values(
        ["HARPNUM", "file_time"]
    ).reset_index(drop=True)

    return index


# ============================================================
# FIND NEAREST MAGNETOGRAM
# ============================================================

def find_nearest_file(
    observation_time,
    harp_files
):

    if harp_files.empty:
        return None, None

    time_differences = (
        harp_files["file_time"]
        - observation_time
    ).abs()

    nearest_index = time_differences.idxmin()

    nearest_row = harp_files.loc[
        nearest_index
    ]

    difference_minutes = (
        abs(
            nearest_row["file_time"]
            - observation_time
        ).total_seconds()
        / 60
    )

    return (
        nearest_row["file_path"],
        difference_minutes
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MAGNETOGRAM LABEL PATH RECONSTRUCTION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD LABELS
    # --------------------------------------------------------

    print()
    print("Loading labels...")

    labels = pd.read_csv(
        LABELS_PATH
    )

    print(
        f"Total labels: {len(labels)}"
    )

    labels["observation_time"] = pd.to_datetime(
        labels["observation_time"],
        errors="coerce"
    )

    labels["HARPNUM"] = pd.to_numeric(
        labels["HARPNUM"],
        errors="coerce"
    )

    labels = labels.dropna(
        subset=[
            "HARPNUM",
            "observation_time",
            "target_24h"
        ]
    ).copy()

    labels["HARPNUM"] = (
        labels["HARPNUM"]
        .astype(int)
    )

    print(
        f"Valid labels: {len(labels)}"
    )

    # --------------------------------------------------------
    # INDEX FILES
    # --------------------------------------------------------

    file_index = build_file_index()

    # --------------------------------------------------------
    # GROUP FILES BY HARPNUM
    # --------------------------------------------------------

    grouped_files = {
        harpnum: group.reset_index(drop=True)
        for harpnum, group
        in file_index.groupby("HARPNUM")
    }

    # --------------------------------------------------------
    # MATCH LABELS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MATCHING LABELS TO NEAREST MAGNETOGRAM")
    print("=" * 60)

    matched_paths = []
    time_differences = []

    for i, row in labels.iterrows():

        harpnum = row["HARPNUM"]

        observation_time = (
            row["observation_time"]
        )

        if harpnum not in grouped_files:

            matched_paths.append(None)
            time_differences.append(None)

            continue

        harp_files = grouped_files[
            harpnum
        ]

        path, difference = find_nearest_file(
            observation_time,
            harp_files
        )

        if (
            difference is not None
            and difference
            <= MAX_TIME_DIFFERENCE_MINUTES
        ):

            matched_paths.append(path)
            time_differences.append(
                difference
            )

        else:

            matched_paths.append(None)
            time_differences.append(
                difference
            )

        if (
            (i + 1) % 500 == 0
        ):

            print(
                f"Processed {i + 1}/{len(labels)}"
            )

    labels["file_path"] = (
        matched_paths
    )

    labels["time_difference_minutes"] = (
        time_differences
    )

    labels["file_exists"] = (
        labels["file_path"]
        .notna()
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MATCHING RESULTS")
    print("=" * 60)

    total_found = (
        labels["file_exists"]
        .sum()
    )

    total_missing = (
        len(labels)
        - total_found
    )

    print(
        f"Matched files : {total_found}"
    )

    print(
        f"Unmatched     : {total_missing}"
    )

    print()
    print("MATCHING BY CLASS")

    class_summary = (
        labels
        .groupby("target_24h")
        ["file_exists"]
        .agg(
            total="count",
            found="sum"
        )
    )

    class_summary["missing"] = (
        class_summary["total"]
        - class_summary["found"]
    )

    print()
    print(
        class_summary
    )

    # --------------------------------------------------------
    # TIME DIFFERENCE STATISTICS
    # --------------------------------------------------------

    matched = labels[
        labels["file_exists"]
    ].copy()

    if len(matched) > 0:

        print()
        print("=" * 60)
        print("TIME DIFFERENCE STATISTICS")
        print("=" * 60)

        print(
            matched[
                "time_difference_minutes"
            ].describe()
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    labels.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Saved corrected labels to:"
    )

    print(
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # FINAL CLASS COUNTS
    # --------------------------------------------------------

    final = labels[
        labels["file_exists"]
    ]

    print()
    print("FINAL USABLE DATASET")

    print(
        final["target_24h"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()