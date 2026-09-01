# ============================================================
# REAL-TIME SOLAR FLARE FEATURE BUILDER
#
# Builds the exact 71 features required by:
#
# xgboost_2012_temporal_features.joblib
#
# Required SHARP parameters:
#
# USFLUX
# TOTUSJH
# TOTPOT
# MEANPOT
# MEANSHR
#
# Temporal windows:
#
# 1 hour
# 3 hours
# 6 hours
# 12 hours
# ============================================================


import numpy as np
import pandas as pd


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURES = [

    "USFLUX",

    "TOTUSJH",

    "TOTPOT",

    "MEANPOT",

    "MEANSHR"

]


# ============================================================
# TEMPORAL WINDOWS
# ============================================================

TIME_WINDOWS = [

    1,

    3,

    6,

    12

]


# ============================================================
# SAFE RELATIVE CHANGE
# ============================================================

def safe_relative_change(

    current,

    previous

):


    if pd.isna(
        current
    ):

        return np.nan


    if pd.isna(
        previous
    ):

        return np.nan


    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    if previous == 0:

        return np.nan


    return (

        current
        - previous

    ) / (

        abs(
            previous
        )

        + 1e-10

    )


# ============================================================
# GET HISTORICAL VALUE
# ============================================================

def get_historical_value(

    history,

    current_time,

    feature,

    hours_back

):


    target_time = (

        current_time

        - pd.Timedelta(

            hours=hours_back

        )

    )


    # --------------------------------------------------------
    # History before or at target time
    # --------------------------------------------------------

    historical = (

        history[

            history[
                "observation_time"
            ]

            <= target_time

        ]

    )


    if len(
        historical
    ) == 0:


        return np.nan


    # --------------------------------------------------------
    # Use latest available observation
    # --------------------------------------------------------

    row = (

        historical

        .sort_values(

            "observation_time"

        )

        .iloc[
            -1
        ]

    )


    return row[
        feature
    ]


# ============================================================
# CALCULATE ROLLING STANDARD DEVIATION
# ============================================================

def calculate_rolling_std(

    history,

    current_time,

    feature,

    hours

):


    start_time = (

        current_time

        - pd.Timedelta(

            hours=hours

        )

    )


    window = (

        history[

            (

                history[
                    "observation_time"
                ]

                >= start_time

            )

            &

            (

                history[
                    "observation_time"
                ]

                <= current_time

            )

        ]

    )


    if len(
        window
    ) < 2:


        return np.nan


    return float(

        window[
            feature
        ]

        .std()

    )


# ============================================================
# BUILD REAL-TIME FEATURE ROW
# ============================================================

def build_realtime_features(

    history

):


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(

        history,

        pd.DataFrame

    ):


        raise TypeError(

            "history must be a pandas DataFrame."

        )


    required_columns = [

        "observation_time"

    ] + BASE_FEATURES


    missing_columns = [

        column

        for column in required_columns

        if column not in history.columns

    ]


    if missing_columns:


        raise ValueError(

            "Missing required columns:\n"

            f"{missing_columns}"

        )


    # --------------------------------------------------------
    # Copy history
    # --------------------------------------------------------

    history = history.copy()


    # --------------------------------------------------------
    # Ensure datetime
    # --------------------------------------------------------

    history[
        "observation_time"
    ] = pd.to_datetime(

        history[
            "observation_time"
        ]

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


    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    current_row = (

        history.iloc[
            -1
        ]

    )


    current_time = (

        current_row[
            "observation_time"
        ]

    )


    # --------------------------------------------------------
    # Output feature dictionary
    # --------------------------------------------------------

    features = {}


    # ========================================================
    # CURRENT BASE FEATURES
    # ========================================================

    for feature in BASE_FEATURES:


        features[
            feature
        ] = (

            current_row[
                feature
            ]

        )


    # ========================================================
    # LOG FEATURES
    # ========================================================

    LOG_FEATURES = [

        "USFLUX",

        "TOTUSJH",

        "TOTPOT",

        "MEANPOT"

    ]


    for feature in LOG_FEATURES:


        value = (

            current_row[
                feature
            ]

        )


        if pd.isna(
            value
        ):


            features[
                f"LOG_{feature}"
            ] = np.nan


        else:


            features[
                f"LOG_{feature}"
            ] = np.log1p(

                max(
                    float(
                        value
                    ),

                    0.0

                )

            )


    # ========================================================
    # TIME FEATURES
    # ========================================================

    features[
        "observation_hour"
    ] = (

        current_time.hour

    )


    features[
        "day_of_year"
    ] = (

        current_time.dayofyear

    )


    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    for feature in BASE_FEATURES:


        current_value = (

            current_row[
                feature
            ]

        )


        for hours in TIME_WINDOWS:


            # ------------------------------------------------
            # Historical value
            # ------------------------------------------------

            previous_value = (

                get_historical_value(

                    history=history,

                    current_time=current_time,

                    feature=feature,

                    hours_back=hours

                )

            )


            # ------------------------------------------------
            # Change
            # ------------------------------------------------

            change = (

                current_value

                - previous_value

                if not pd.isna(
                    previous_value
                )

                else np.nan

            )


            features[

                f"{feature}_CHANGE_{hours}h"

            ] = change


            # ------------------------------------------------
            # Relative change
            # ------------------------------------------------

            features[

                f"{feature}_RELCHANGE_{hours}h"

            ] = (

                safe_relative_change(

                    current=current_value,

                    previous=previous_value

                )

            )


            # ------------------------------------------------
            # Rolling standard deviation
            # ------------------------------------------------

            features[

                f"{feature}_ROLLSTD_{hours}h"

            ] = (

                calculate_rolling_std(

                    history=history,

                    current_time=current_time,

                    feature=feature,

                    hours=hours

                )

            )


    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    feature_df = pd.DataFrame(

        [

            features

        ]

    )


    return feature_df