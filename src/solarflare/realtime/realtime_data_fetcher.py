# ============================================================
# REAL-TIME SOLAR DATA FETCHER
#
# CURRENT VERSION:
#
# Uses historical processed SHARP data to simulate a
# real-time data stream.
#
# This allows us to test the complete real-time pipeline.
#
# LATER:
#
# The data loading section can be replaced with an API-based
# SHARP/HMI data fetcher without changing the rest of the
# prediction system.
# ============================================================


import sys

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


SRC_PATH = (
    PROJECT_ROOT
    / "src"
)


if str(
    SRC_PATH
) not in sys.path:

    sys.path.insert(

        0,

        str(
            SRC_PATH
        )

    )


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (

    PROJECT_ROOT

    / "data"

    / "processed"

    / "features"

    / "sharp_goes_temporal_features_2012_full.parquet"

)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_data():

    print()

    print(
        "Loading solar observation data..."
    )


    if not DATA_PATH.exists():

        raise FileNotFoundError(

            f"Processed dataset not found:\n"
            f"{DATA_PATH}"

        )


    df = pd.read_parquet(

        DATA_PATH

    )


    # --------------------------------------------------------
    # Ensure datetime format
    # --------------------------------------------------------

    df[
        "observation_time"
    ] = pd.to_datetime(

        df[
            "observation_time"
        ]

    )


    print(

        f"Loaded observations: "
        f"{len(df)}"

    )


    return df


# ============================================================
# GET ACTIVE REGION HISTORY
# ============================================================

def get_region_history(

    df,

    harpnum,

    observation_time=None

):

    # --------------------------------------------------------
    # Filter HARPNUM
    # --------------------------------------------------------

    history = (

        df[

            df[
                "HARPNUM"
            ]

            == harpnum

        ]

        .copy()

    )


    if len(
        history
    ) == 0:

        raise ValueError(

            f"No observations found for "
            f"HARPNUM {harpnum}"

        )


    # --------------------------------------------------------
    # Optional time cutoff
    #
    # Important for simulating real-time prediction.
    #
    # We must only use observations that happened
    # BEFORE or AT the requested observation time.
    # --------------------------------------------------------

    if observation_time is not None:


        observation_time = pd.to_datetime(

            observation_time

        )


        history = (

            history[

                history[
                    "observation_time"
                ]

                <= observation_time

            ]

            .copy()

        )


    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    history = (

        history

        .sort_values(

            "observation_time"

        )

        .reset_index(

            drop=True

        )

    )


    if len(
        history
    ) == 0:

        raise ValueError(

            f"No historical observations available "
            f"before {observation_time} for "
            f"HARPNUM {harpnum}"

        )


    return history


# ============================================================
# GET LATEST OBSERVATION
# ============================================================

def get_latest_observation(

    df,

    harpnum

):

    history = (

        get_region_history(

            df=df,

            harpnum=harpnum

        )

    )


    latest = (

        history.iloc[
            -1
        ]

    )


    return latest


# ============================================================
# FIND TEST REGION
#
# Finds a HARPNUM with enough history for the temporal
# features.
# ============================================================

def find_test_region(

    df,

    minimum_observations=20

):

    counts = (

        df
        .groupby(
            "HARPNUM"
        )
        .size()

    )


    valid_regions = (

        counts[

            counts
            >= minimum_observations

        ]

    )


    if len(
        valid_regions
    ) == 0:

        raise RuntimeError(

            "No HARPNUM has enough observation "
            "history."

        )


    harpnum = (

        valid_regions
        .index[
            -1
        ]

    )


    return harpnum


# ============================================================
# GET SIMULATED REAL-TIME INPUT
#
# Returns:
#
# latest observation
# +
# historical observations available up to that point
# ============================================================

def get_realtime_input(

    df,

    harpnum=None

):

    # --------------------------------------------------------
    # Automatically select a test region
    # --------------------------------------------------------

    if harpnum is None:


        harpnum = (

            find_test_region(
                df
            )

        )


    # --------------------------------------------------------
    # Get complete history
    # --------------------------------------------------------

    full_history = (

        get_region_history(

            df=df,

            harpnum=harpnum

        )

    )


    # --------------------------------------------------------
    # Latest observation simulates the current
    # real-time observation.
    # --------------------------------------------------------

    latest_observation = (

        full_history.iloc[
            -1
        ]

    )


    observation_time = (

        latest_observation[
            "observation_time"
        ]

    )


    # --------------------------------------------------------
    # Get only information available at that time.
    # --------------------------------------------------------

    history = (

        get_region_history(

            df=df,

            harpnum=harpnum,

            observation_time=observation_time

        )

    )


    return (

        history,

        latest_observation

    )


# ============================================================
# QUICK TEST
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )


    print(
        " REAL-TIME DATA FETCHER TEST"
    )


    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = (

        load_data()

    )


    # --------------------------------------------------------
    # Get simulated real-time input
    # --------------------------------------------------------

    history, latest = (

        get_realtime_input(
            df
        )

    )


    print()

    print(
        "=" * 60
    )


    print(
        " SIMULATED REAL-TIME OBSERVATION"
    )


    print(
        "=" * 60
    )


    print()

    print(

        f"HARPNUM: "
        f"{latest['HARPNUM']}"

    )


    print(

        f"Observation time: "
        f"{latest['observation_time']}"

    )


    print()

    print(

        f"Available historical observations: "
        f"{len(history)}"

    )


    print()

    print(
        "=" * 60
    )


    print(
        " REAL-TIME DATA FETCHER TEST COMPLETE"
    )


    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()