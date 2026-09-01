from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SHARP_PATH = Path(
    "data/raw/sharp/"
    "sharp_2012_full_year.parquet"
)

GOES_PATH = Path(
    "data/raw/goes/"
    "goes_flares_2012.csv"
)

OUTPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_full.parquet"
)
FORECAST_HOURS = 24


# ============================================================
# LOAD SHARP
# ============================================================

def load_sharp() -> pd.DataFrame:
    """Load and prepare SHARP observations."""

    print("Loading SHARP data...")

    sharp = pd.read_parquet(
        SHARP_PATH
    ).copy()

    sharp["observation_time"] = pd.to_datetime(
        sharp["T_REC"],
        format="%Y.%m.%d_%H:%M:%S_TAI",
        errors="coerce",
    )

    sharp["NOAA_AR"] = pd.to_numeric(
        sharp["NOAA_AR"],
        errors="coerce",
    )

    # Remove observations without a valid
    # NOAA active-region association.
    sharp = sharp[
        sharp["NOAA_AR"].notna()
        & (sharp["NOAA_AR"] != 0)
    ].copy()

    sharp = sharp.dropna(
        subset=["observation_time"]
    )

    sharp = sharp.sort_values(
        [
            "NOAA_AR",
            "observation_time",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"SHARP rows: {len(sharp)}"
    )

    print(
        f"SHARP active regions: "
        f"{sharp['NOAA_AR'].nunique()}"
    )

    return sharp


# ============================================================
# LOAD GOES
# ============================================================

def load_goes() -> pd.DataFrame:
    """Load and prepare GOES M/X flare events."""

    print()
    print("Loading GOES data...")

    goes = pd.read_csv(
        GOES_PATH
    ).copy()

    goes["start_time"] = pd.to_datetime(
        goes["start_time"],
        errors="coerce",
    )

    goes["active_region"] = pd.to_numeric(
        goes["active_region"],
        errors="coerce",
    )

    # GOES active-region numbering is mapped
    # to the NOAA_AR representation used by SHARP.
    goes["NOAA_AR"] = (
        goes["active_region"] + 10000
    )

    # Keep only M/X flares.
    goes = goes[
        goes["flare_class"]
        .astype(str)
        .str.startswith(
            ("M", "X")
        )
    ].copy()

    goes["flare_time"] = (
        goes["start_time"]
    )

    goes = goes.dropna(
        subset=[
            "flare_time",
            "NOAA_AR",
        ]
    )

    goes = goes.sort_values(
        [
            "NOAA_AR",
            "flare_time",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"GOES M/X events: {len(goes)}"
    )

    print(
        f"GOES active regions: "
        f"{goes['NOAA_AR'].nunique()}"
    )

    print()
    print("Sample M/X events:")

    print(
        goes[
            [
                "flare_time",
                "flare_class",
                "active_region",
                "NOAA_AR",
            ]
        ].head()
    )

    return goes


# ============================================================
# CREATE 24-HOUR LABELS
# ============================================================

def create_labels(
    sharp: pd.DataFrame,
    goes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign target_24h = 1 when an M/X flare
    begins within the next 24 hours for the
    same active region.
    """

    print()
    print(
        "========== CREATING LABELS =========="
    )

    sharp = sharp.copy()

    # Store flare times by active region.
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

        observation_time = (
            row["observation_time"]
        )

        if pd.isna(region) or pd.isna(
            observation_time
        ):
            labels.append(0)
            continue

        window_end = (
            observation_time
            + pd.Timedelta(
                hours=FORECAST_HOURS
            )
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


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    data: pd.DataFrame,
) -> None:

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
        f"SHARP observations: "
        f"{len(data)}"
    )

    print(
        f"Active regions: "
        f"{data['NOAA_AR'].nunique()}"
    )

    print()

    print(
        "Time range:"
    )

    print(
        f"{data['observation_time'].min()} "
        f"→ "
        f"{data['observation_time'].max()}"
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
        f"Positive observations: "
        f"{int(data['target_24h'].sum())}"
    )


# ============================================================
# SAVE
# ============================================================

def save_data(
    data: pd.DataFrame,
) -> None:

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "========================================"
    )

    print(
        " SHARP + GOES ALIGNMENT"
    )

    print(
        "========================================"
    )

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
        "========== ALIGNMENT COMPLETE =========="
    )


if __name__ == "__main__":
    main()