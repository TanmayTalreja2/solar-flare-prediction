from pathlib import Path

import pandas as pd


SHARP_PATH = Path(
    "data/raw/sharp/sharp_sample_2012_03_07.parquet"
)

GOES_PATH = Path(
    "data/processed/goes/goes_flares_2012.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/aligned/"
    "sharp_goes_training_2012_03_07.parquet"
)


SHARP_FEATURES = [
    "USFLUX",
    "TOTUSJH",
    "TOTPOT",
    "MEANPOT",
    "MEANSHR",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SHARP and GOES datasets."""

    sharp = pd.read_parquet(
        SHARP_PATH
    )

    goes = pd.read_parquet(
        GOES_PATH
    )

    return sharp, goes


def prepare_sharp(
    sharp: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare SHARP observations."""

    sharp = sharp.copy()

    # SHARP uses the format:
    # 2012.03.07_00:00:00_TAI
    sharp["observation_time"] = pd.to_datetime(
        sharp["T_REC"],
        format="%Y.%m.%d_%H:%M:%S_TAI",
        errors="coerce",
    )

    sharp["NOAA_AR"] = pd.to_numeric(
        sharp["NOAA_AR"],
        errors="coerce",
    )

    return sharp

def prepare_goes(
    goes: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare GOES flare events."""

    goes = goes.copy()

    goes["peak_time"] = pd.to_datetime(
        goes["peak_time"],
        errors="coerce",
    )

    goes["active_region"] = pd.to_numeric(
        goes["active_region"],
        errors="coerce",
    )

    # GOES stores the NOAA active-region number
    # without the leading 11 for this historical data.
    #
    # Example:
    #
    # GOES: 1429
    # SHARP: 11429
    #
    # Therefore convert GOES IDs into the SHARP/NOAA format.

    goes["NOAA_AR"] = (
        goes["active_region"] + 10000
    )

    # Keep only M/X flares.
    goes = goes[
        goes["flare_category"].isin(
            ["M", "X"]
        )
    ].copy()

    return goes


def create_labels(
    sharp: pd.DataFrame,
    goes: pd.DataFrame,
) -> pd.DataFrame:
    """Create 24-hour M/X flare labels."""

    results = []

    for _, observation in sharp.iterrows():

        observation_time = (
            observation["observation_time"]
        )

        region = observation["NOAA_AR"]

        window_end = (
            observation_time
            + pd.Timedelta(hours=24)
        )

        matching_flares = goes[
    (goes["NOAA_AR"] == region)
    & (
        goes["peak_time"]
        > observation_time
    )
    & (
        goes["peak_time"]
        <= window_end
    )
]

        row = observation.to_dict()

        if len(matching_flares) > 0:

            first_flare = (
                matching_flares
                .sort_values("peak_time")
                .iloc[0]
            )

            row["target_24h"] = 1
            row["next_flare_class"] = (
                first_flare["flare_class"]
            )
            row["next_flare_time"] = (
                first_flare["peak_time"]
            )

        else:

            row["target_24h"] = 0
            row["next_flare_class"] = None
            row["next_flare_time"] = pd.NaT

        results.append(row)

    return pd.DataFrame(results)


def save_dataset(
    data: pd.DataFrame,
) -> None:
    """Save aligned training dataset."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========== ALIGNMENT COMPLETE ==========")
    print(f"Rows: {len(data)}")
    print(f"Output: {OUTPUT_PATH}")

    print()
    print("========== TARGET DISTRIBUTION ==========")
    print(
        data["target_24h"]
        .value_counts()
    )


def main() -> None:
    """Run SHARP-GOES alignment."""

    print("Loading datasets...")

    sharp, goes = load_data()

    print(f"SHARP rows: {len(sharp)}")
    print(f"GOES rows: {len(goes)}")

    sharp = prepare_sharp(
        sharp
    )

    goes = prepare_goes(
        goes
    )

    print(
        f"M/X GOES events: {len(goes)}"
    )

    aligned = create_labels(
        sharp,
        goes,
    )

    save_dataset(
        aligned
    )


if __name__ == "__main__":
    main()