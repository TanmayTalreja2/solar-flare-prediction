from pathlib import Path

import pandas as pd


SHARP_PATH = Path(
    "data/raw/sharp/"
    "sharp_2012_03_01_30d.parquet"
)

GOES_PATH = Path(
    "data/raw/goes/"
    "goes_flares_2012.csv"
)

OUTPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_03.parquet"
)


def load_sharp() -> pd.DataFrame:
    """Load and prepare SHARP observations."""

    print("Loading SHARP data...")

    sharp = pd.read_parquet(
        SHARP_PATH
    )

    sharp = sharp.copy()

    sharp["observation_time"] = pd.to_datetime(
        sharp["T_REC"],
        format="%Y.%m.%d_%H:%M:%S_TAI",
        errors="coerce",
    )

    sharp["NOAA_AR"] = pd.to_numeric(
        sharp["NOAA_AR"],
        errors="coerce",
    )

    print(
        f"SHARP rows: {len(sharp)}"
    )

    print(
        f"SHARP active regions: "
        f"{sharp['NOAA_AR'].nunique()}"
    )

    return sharp


def load_goes() -> pd.DataFrame:
    """Load and prepare GOES flare events."""

    print()
    print("Loading GOES data...")

    goes = pd.read_csv(
        GOES_PATH
    )

    goes = goes.copy()

    # Convert timestamps.
    goes["start_time"] = pd.to_datetime(
        goes["start_time"],
        errors="coerce",
    )

    goes["end_time"] = pd.to_datetime(
        goes["end_time"],
        errors="coerce",
    )

    goes["time"] = pd.to_datetime(
        goes["time"],
        errors="coerce",
    )

    # Active region numbers are numeric in GOES.
    goes["active_region"] = pd.to_numeric(
        goes["active_region"],
        errors="coerce",
    )
    goes["NOAA_AR"] = (
    goes["active_region"] + 10000
    )

    # Keep only M and X class flares.
    goes = goes[
        goes["flare_class"]
        .astype(str)
        .str.startswith(("M", "X"))
    ].copy()

    # We use flare START time for the future-event label.
    goes["flare_time"] = goes["start_time"]

    goes = goes.dropna(
        subset=[
            "flare_time",
            "active_region",
        ]
    )

    goes = goes.sort_values(
        "flare_time"
    )

    print(
        f"GOES M/X events: {len(goes)}"
    )

    print(
        f"GOES active regions: "
        f"{goes['active_region'].nunique()}"
    )

    print()
    print("Sample M/X events:")

    print(
        goes[
            [
                "flare_time",
                "flare_class",
                "active_region",
            ]
        ].head()
    )

    return goes



def create_labels(
        
    sharp: pd.DataFrame,
    goes: pd.DataFrame,
) -> pd.DataFrame:
    sharp = sharp[
    sharp["NOAA_AR"].notna()
    & (sharp["NOAA_AR"] != 0)
].copy()
    """Assign a 24-hour future M/X flare label."""

    print()
    print(
        "========== CREATING LABELS =========="
    )

    sharp = sharp.copy()

    # Prepare dictionary:
    # NOAA_AR -> sorted M/X flare times
    flare_times = {}

    for region, group in goes.groupby(
        "NOAA_AR"
    ):
        flare_times[region] = (
            group["flare_time"]
            .sort_values()
            .tolist()
        )

    labels = []

    for _, row in sharp.iterrows():

        region = row["NOAA_AR"]
        observation_time = row[
            "observation_time"
        ]

        if pd.isna(region) or pd.isna(
            observation_time
        ):
            labels.append(0)
            continue

        window_end = (
            observation_time
            + pd.Timedelta(hours=24)
        )

        event_times = flare_times.get(
            region,
            [],
        )

        target = any(
            observation_time < flare_time <= window_end
            for flare_time in event_times
        )

        labels.append(
            int(target)
        )

    sharp["target_24h"] = labels

    return sharp


def print_summary(
    data: pd.DataFrame,
) -> None:
    """Print alignment summary."""

    print()
    print(
        "========================================"
    )
    print(
        " ALIGNMENT SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"SHARP observations: {len(data)}"
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        data["target_24h"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Target percentages:"
    )

    print(
        data["target_24h"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print()
    print(
        "Positive observations:"
    )

    print(
        int(
            data["target_24h"].sum()
        )
    )


def save_data(
    data: pd.DataFrame,
) -> None:
    """Save the aligned dataset."""

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
        "Dataset saved:"
    )

    print(
        OUTPUT_PATH
    )


def main() -> None:
    """Run the complete 30-day alignment pipeline."""

    sharp = load_sharp()

    goes = load_goes()

    aligned = create_labels(
        sharp,
        goes,
    )

    print_summary(
        aligned
    )

    save_data(
        aligned
    )

    print()
    print(
        "========== 30-DAY ALIGNMENT COMPLETE =========="
    )


if __name__ == "__main__":
    main()