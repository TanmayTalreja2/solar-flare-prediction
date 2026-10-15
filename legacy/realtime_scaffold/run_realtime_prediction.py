# ============================================================
# REAL-TIME HYBRID SOLAR FLARE PREDICTION
#
# Uses:
#
#   1. Historical SHARP observations for simulated realtime data
#   2. Real-time temporal feature builder
#   3. XGBoost tabular model
#   4. ResNet18 magnetogram model
#   5. Hybrid ensemble
#
# Hybrid Ensemble:
#
#   20% XGBoost
#   80% Magnetogram ResNet18
#
# IMPORTANT:
#
# For real-time inference we DO NOT know target_24h.
#
# Therefore, this script searches for the most recent
# available magnetogram at or before the observation time.
# ============================================================


import sys

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import torch
import torch.nn as nn

import torchvision.models as models


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
# IMPORT REAL-TIME FEATURE BUILDER
# ============================================================

from solarflare.realtime.feature_builder import (

    build_realtime_features

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


MAGNETOGRAM_DIR = (

    PROJECT_ROOT
    / "data"
    / "processed"
    / "magnetograms"

)


XGB_MODEL_PATH = (

    PROJECT_ROOT
    / "models"
    / "xgboost_2012_temporal_features.joblib"

)


MAGNETOGRAM_MODEL_PATH = (

    PROJECT_ROOT
    / "models"
    / "cnn_magnetogram.pt"

)


# ============================================================
# ENSEMBLE SETTINGS
# ============================================================

XGB_WEIGHT = 0.2


MAGNETOGRAM_WEIGHT = 0.8


# ============================================================
# REAL-TIME SETTINGS
# ============================================================

MAX_MAGNETOGRAM_AGE_HOURS = 12


# ============================================================
# TEST PERIOD
#
# Historical data before this was used for training.
#
# We simulate realtime predictions after this date.
# ============================================================

TEST_START = pd.Timestamp(

    "2012-07-01 00:00:00"

)


# ============================================================
# BUILD MAGNETOGRAM RESNET18
# ============================================================

def build_magnetogram_model():


    print(
        "Building ResNet18 magnetogram model..."
    )


    # --------------------------------------------------------
    # Base ResNet18
    # --------------------------------------------------------

    model = models.resnet18(

        weights=None

    )


    # --------------------------------------------------------
    # Magnetograms are grayscale
    #
    # ResNet normally expects RGB:
    #
    # [3, H, W]
    #
    # Our magnetograms:
    #
    # [1, H, W]
    # --------------------------------------------------------

    model.conv1 = nn.Conv2d(

        in_channels=1,

        out_channels=64,

        kernel_size=7,

        stride=2,

        padding=3,

        bias=False

    )


    # --------------------------------------------------------
    # Classification layer
    #
    # Checkpoint contains:
    #
    # fc.1.weight
    # fc.1.bias
    #
    # Therefore architecture is:
    #
    # Sequential(
    #     Dropout,
    #     Linear
    # )
    # --------------------------------------------------------

    num_features = (

        model.fc.in_features

    )


    model.fc = nn.Sequential(

        nn.Dropout(

            p=0.3

        ),

        nn.Linear(

            num_features,

            1

        )

    )


    return model


# ============================================================
# HYBRID SOLAR FLARE PREDICTOR
# ============================================================

class HybridSolarFlarePredictor:


    def __init__(

        self,

        xgb_model_path=XGB_MODEL_PATH,

        magnetogram_model_path=MAGNETOGRAM_MODEL_PATH

    ):


        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )


        print()

        print(
            "=" * 60
        )


        print(
            " LOADING HYBRID SOLAR FLARE MODELS"
        )


        print(
            "=" * 60
        )


        print(

            f"Using device: "
            f"{self.device}"

        )


        # ====================================================
        # LOAD XGBOOST
        # ====================================================

        print()

        print(
            "Loading XGBoost model..."
        )


        xgb_model_path = Path(

            xgb_model_path

        )


        if not xgb_model_path.exists():


            raise FileNotFoundError(

                f"XGBoost model not found:\n"
                f"{xgb_model_path}"

            )


        package = joblib.load(

            xgb_model_path

        )


        self.xgb_model = (

            package[
                "model"
            ]

        )


        self.imputer = (

            package[
                "imputer"
            ]

        )


        self.features = (

            package[
                "features"
            ]

        )


        print(
            "XGBoost loaded successfully."
        )


        # ====================================================
        # LOAD MAGNETOGRAM RESNET18
        # ====================================================

        print()

        print(
            "Loading magnetogram ResNet18..."
        )


        magnetogram_model_path = Path(

            magnetogram_model_path

        )


        if not magnetogram_model_path.exists():


            raise FileNotFoundError(

                f"Magnetogram model not found:\n"
                f"{magnetogram_model_path}"

            )


        self.magnetogram_model = (

            build_magnetogram_model()

        )


        checkpoint = torch.load(

            magnetogram_model_path,

            map_location=self.device

        )


        # ----------------------------------------------------
        # Handle checkpoint formats
        # ----------------------------------------------------

        if isinstance(

            checkpoint,

            dict

        ):


            if (

                "model_state_dict"

                in checkpoint

            ):


                state_dict = (

                    checkpoint[
                        "model_state_dict"
                    ]

                )


            elif (

                "state_dict"

                in checkpoint

            ):


                state_dict = (

                    checkpoint[
                        "state_dict"
                    ]

                )


            else:


                state_dict = checkpoint


        else:


            state_dict = checkpoint


        # ----------------------------------------------------
        # Remove DataParallel prefix
        # ----------------------------------------------------

        cleaned_state_dict = {}


        for key, value in state_dict.items():


            if key.startswith(

                "module."

            ):


                key = key.replace(

                    "module.",

                    "",

                    1

                )


            cleaned_state_dict[
                key
            ] = value


        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        self.magnetogram_model.load_state_dict(

            cleaned_state_dict,

            strict=True

        )


        self.magnetogram_model = (

            self.magnetogram_model.to(

                self.device

            )

        )


        self.magnetogram_model.eval()


        print(
            "Magnetogram ResNet18 loaded successfully."
        )


        print()

        print(
            "=" * 60
        )


        print(
            " HYBRID SYSTEM READY"
        )


        print(
            "=" * 60
        )


    # ========================================================
    # XGBOOST PREDICTION
    # ========================================================

    def predict_xgb(

        self,

        feature_row

    ):


        # ----------------------------------------------------
        # Convert Series to DataFrame
        # ----------------------------------------------------

        if isinstance(

            feature_row,

            pd.Series

        ):


            feature_row = (

                feature_row

                .to_frame()

                .T

            )


        # ----------------------------------------------------
        # Check required features
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in self.features

            if feature not in feature_row.columns

        ]


        if missing_features:


            raise ValueError(

                f"Missing XGBoost features:\n"
                f"{missing_features}"

            )


        # ----------------------------------------------------
        # Exact feature order
        # ----------------------------------------------------

        X = (

            feature_row[
                self.features
            ]

            .copy()

        )


        # ----------------------------------------------------
        # Apply fitted imputer
        # ----------------------------------------------------

        X = (

            self.imputer.transform(

                X

            )

        )


        # ----------------------------------------------------
        # Predict probability
        # ----------------------------------------------------

        probability = (

            self.xgb_model

            .predict_proba(

                X

            )[:, 1][0]

        )


        return float(

            probability

        )


    # ========================================================
    # MAGNETOGRAM PREDICTION
    # ========================================================

    def predict_magnetogram(

        self,

        magnetogram

    ):


        # ----------------------------------------------------
        # Accept:
        #
        # 1. NumPy array
        #
        # OR
        #
        # 2. Path to NPZ magnetogram
        # ----------------------------------------------------

        if isinstance(

            magnetogram,

            (

                str,

                Path

            )

        ):


            magnetogram_path = Path(

                magnetogram

            )


            if not magnetogram_path.exists():


                raise FileNotFoundError(

                    f"Magnetogram file not found:\n"
                    f"{magnetogram_path}"

                )


            magnetogram = np.load(

                magnetogram_path

            )[
                "img"
            ]


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        magnetogram = np.asarray(

            magnetogram

        )


        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if magnetogram.ndim != 2:


            raise ValueError(

                "Magnetogram must be a 2D image.\n"

                f"Received shape: "
                f"{magnetogram.shape}"

            )


        # ----------------------------------------------------
        # Convert:
        #
        # [H, W]
        #
        # into:
        #
        # [Batch, Channel, H, W]
        # ----------------------------------------------------

        tensor = torch.tensor(

            magnetogram,

            dtype=torch.float32

        )


        tensor = tensor.unsqueeze(

            0

        ).unsqueeze(

            0

        )


        tensor = tensor.to(

            self.device

        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():


            logits = (

                self.magnetogram_model(

                    tensor

                )

            )


            probability = (

                torch.sigmoid(

                    logits

                )

                .squeeze()

                .cpu()

                .item()

            )


        return float(

            probability

        )


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    def classify_risk(

        self,

        probability

    ):


        if probability < 0.20:


            return "LOW RISK"


        elif probability < 0.40:


            return "MODERATE RISK"


        elif probability < 0.70:


            return "HIGH RISK"


        else:


            return "VERY HIGH RISK"


    # ========================================================
    # HYBRID PREDICTION
    # ========================================================

    def predict(

        self,

        feature_row,

        magnetogram

    ):


        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        xgb_probability = (

            self.predict_xgb(

                feature_row

            )

        )


        # ----------------------------------------------------
        # MAGNETOGRAM RESNET18
        # ----------------------------------------------------

        magnetogram_probability = (

            self.predict_magnetogram(

                magnetogram

            )

        )


        # ----------------------------------------------------
        # HYBRID ENSEMBLE
        # ----------------------------------------------------

        ensemble_probability = (

            XGB_WEIGHT

            * xgb_probability

            +

            MAGNETOGRAM_WEIGHT

            * magnetogram_probability

        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        risk_level = (

            self.classify_risk(

                ensemble_probability

            )

        )


        return {


            "xgb_probability":

                xgb_probability,


            "magnetogram_probability":

                magnetogram_probability,


            "ensemble_probability":

                ensemble_probability,


            "risk_level":

                risk_level

        }


# ============================================================
# FIND NEAREST AVAILABLE MAGNETOGRAM
#
# Real-time safe:
#
# Only uses magnetograms whose timestamp is:
#
# magnetogram_time <= observation_time
#
# This prevents future data leakage.
# ============================================================

def find_nearest_magnetogram(

    harpnum,

    observation_time,

    magnetogram_dir,

    max_time_difference_hours=12

):


    print()

    print(
        "Searching for nearest available magnetogram..."
    )


    magnetogram_dir = Path(

        magnetogram_dir

    )


    if not magnetogram_dir.exists():


        raise FileNotFoundError(

            f"Magnetogram directory not found:\n"
            f"{magnetogram_dir}"

        )


    # --------------------------------------------------------
    # Search files belonging to this HARPNUM
    #
    # Example:
    #
    # harp_2344_20121231_212400_t0.npz
    # --------------------------------------------------------

    pattern = (

        f"harp_{int(harpnum)}_*.npz"

    )


    candidate_files = list(

        magnetogram_dir.glob(

            pattern

        )

    )


    if len(candidate_files) == 0:


        raise FileNotFoundError(

            f"No magnetograms found for "
            f"HARPNUM {harpnum}"

        )


    print(

        f"Found {len(candidate_files)} "
        f"candidate magnetograms."

    )


    candidates = []


    # --------------------------------------------------------
    # Parse timestamps
    # ----------------------------------------------------

    for path in candidate_files:


        filename = (

            path.stem

        )


        # Expected format:
        #
        # harp_2344_20121231_212400_t0
        #
        # Split result:
        #
        # [0] harp
        # [1] 2344
        # [2] 20121231
        # [3] 212400
        # [4] t0
        # ----------------------------------------------------

        parts = filename.split(

            "_"

        )


        if len(parts) < 5:


            continue


        try:


            date_part = (

                parts[2]

            )


            time_part = (

                parts[3]

            )


            timestamp_string = (

                f"{date_part}_"
                f"{time_part}"

            )


            magnetogram_time = (

                pd.to_datetime(

                    timestamp_string,

                    format="%Y%m%d_%H%M%S"

                )

            )


        except Exception:


            continue


        candidates.append(

            (

                magnetogram_time,

                path

            )

        )


    # --------------------------------------------------------
    # Safety check
    # ----------------------------------------------------

    if len(candidates) == 0:


        raise RuntimeError(

            f"Could not parse magnetogram "
            f"timestamps for HARPNUM {harpnum}"

        )


    # --------------------------------------------------------
    # REAL-TIME RULE
    #
    # Only use data available at or before
    # the prediction timestamp.
    # ----------------------------------------------------

    past_candidates = [

        candidate

        for candidate in candidates

        if candidate[0] <= observation_time

    ]


    if len(past_candidates) == 0:


        raise FileNotFoundError(

            f"No magnetogram exists at or before:\n"
            f"{observation_time}\n\n"

            f"HARPNUM: "
            f"{harpnum}"

        )


    # --------------------------------------------------------
    # Select latest available magnetogram
    # ----------------------------------------------------

    nearest_time, nearest_path = max(

        past_candidates,

        key=lambda x: x[0]

    )


    # --------------------------------------------------------
    # Calculate age
    # ----------------------------------------------------

    time_difference = (

        observation_time

        -

        nearest_time

    )


    time_difference_hours = (

        time_difference.total_seconds()

        /

        3600

    )


    # --------------------------------------------------------
    # Safety limit
    # ----------------------------------------------------

    if (

        time_difference_hours

        >

        max_time_difference_hours

    ):


        raise RuntimeError(

            f"Nearest magnetogram is too old.\n\n"

            f"Observation time: "
            f"{observation_time}\n"

            f"Magnetogram time: "
            f"{nearest_time}\n"

            f"Difference: "
            f"{time_difference_hours:.2f} hours\n"

            f"Maximum allowed: "
            f"{max_time_difference_hours} hours"

        )


    return (

        nearest_path,

        nearest_time,

        time_difference_hours

    )


# ============================================================
# LOAD REAL-TIME OBSERVATION
#
# Currently uses historical data to simulate realtime input.
#
# Later this function can be replaced with a live SHARP
# data/API fetcher without changing the prediction system.
# ============================================================

def load_simulated_realtime_observation():


    print()

    print(
        "Loading solar observation data..."
    )


    if not DATA_PATH.exists():


        raise FileNotFoundError(

            f"Data file not found:\n"
            f"{DATA_PATH}"

        )


    df = pd.read_parquet(

        DATA_PATH

    )


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


    # --------------------------------------------------------
    # Restrict to test period
    # ----------------------------------------------------

    test_df = (

        df[

            df[
                "observation_time"
            ]

            >= TEST_START

        ]

        .copy()

    )


    if len(test_df) == 0:


        raise RuntimeError(

            "No observations found in "
            "the test period."

        )


    # --------------------------------------------------------
    # Select latest available observation
    # ----------------------------------------------------

    test_df = (

        test_df.sort_values(

            "observation_time"

        )

    )


    latest_row = (

        test_df.iloc[-1]

    )


    return (

        df,

        latest_row

    )


# ============================================================
# GET ACTIVE REGION HISTORY
#
# Only observations up to the prediction time are used.
#
# This prevents future information leakage.
# ============================================================

def get_active_region_history(

    df,

    harpnum,

    observation_time

):


    history = (

        df[

            (

                df[
                    "HARPNUM"
                ]

                == harpnum

            )

            &

            (

                df[
                    "observation_time"
                ]

                <= observation_time

            )

        ]

        .sort_values(

            "observation_time"

        )

        .copy()

    )


    return history


# ============================================================
# MAIN
# ============================================================

def main():


    print()

    print(
        "=" * 60
    )


    print(
        " REAL-TIME HYBRID SOLAR FLARE PREDICTION"
    )


    print(
        "=" * 60
    )


    # ========================================================
    # [1/5] LOAD OBSERVATION DATA
    # ========================================================

    print()

    print(
        "[1/5] Loading solar observation data..."
    )


    df, latest_observation = (

        load_simulated_realtime_observation()

    )


    # ========================================================
    # OBSERVATION INFORMATION
    # ========================================================

    harpnum = (

        latest_observation[
            "HARPNUM"
        ]

    )


    observation_time = (

        latest_observation[
            "observation_time"
        ]

    )


    # ========================================================
    # [2/5] GET ACTIVE REGION HISTORY
    # ========================================================

    print()

    print(
        "[2/5] Getting active region history..."
    )


    history = (

        get_active_region_history(

            df=df,

            harpnum=harpnum,

            observation_time=observation_time

        )

    )


    if len(history) == 0:


        raise RuntimeError(

            f"No historical observations found "
            f"for HARPNUM {harpnum}"

        )


    print()

    print(

        f"HARPNUM: "
        f"{harpnum}"

    )


    print(

        f"Observation time: "
        f"{observation_time}"

    )


    print(

        f"Historical observations available: "
        f"{len(history)}"

    )


    # ========================================================
    # [3/5] BUILD REAL-TIME FEATURES
    # ========================================================

    print()

    print(
        "[3/5] Building real-time features..."
    )


    realtime_features = (

        build_realtime_features(

            history

        )

    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(realtime_features) == 0:


        raise RuntimeError(

            "Feature builder returned "
            "an empty DataFrame."

        )


    print(

        f"Generated features: "
        f"{realtime_features.shape[1]}"

    )


    # ========================================================
    # [4/5] FIND NEAREST MAGNETOGRAM
    # ========================================================

    print()

    print(
        "[4/5] Finding magnetogram..."
    )


    magnetogram_path, magnetogram_time, magnetogram_age = (

        find_nearest_magnetogram(

            harpnum=harpnum,

            observation_time=observation_time,

            magnetogram_dir=MAGNETOGRAM_DIR,

            max_time_difference_hours=(
                MAX_MAGNETOGRAM_AGE_HOURS
            )

        )

    )


    print()

    print(
        "Magnetogram selected:"
    )


    print(

        f"File: "
        f"{magnetogram_path.name}"

    )


    print(

        f"Magnetogram time: "
        f"{magnetogram_time}"

    )


    print(

        f"Age of magnetogram: "
        f"{magnetogram_age:.2f} hours"

    )


    # ========================================================
    # [5/5] RUN HYBRID PREDICTION
    # ========================================================

    print()

    print(
        "[5/5] Running hybrid prediction..."
    )


    predictor = (

        HybridSolarFlarePredictor()

    )


    result = (

        predictor.predict(

            feature_row=realtime_features,

            magnetogram=magnetogram_path

        )

    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()

    print(
        "=" * 60
    )


    print(
        " REAL-TIME HYBRID PREDICTION RESULT"
    )


    print(
        "=" * 60
    )


    print()

    print(

        f"HARPNUM                 : "
        f"{harpnum}"

    )


    print(

        f"Observation Time        : "
        f"{observation_time}"

    )


    print(

        f"Magnetogram Time        : "
        f"{magnetogram_time}"

    )


    print(

        f"Magnetogram Age         : "
        f"{magnetogram_age:.2f} hours"

    )


    print()

    print(

        f"XGBoost Probability     : "
        f"{result['xgb_probability']:.4f}"

    )


    print(

        f"Magnetogram Probability : "
        f"{result['magnetogram_probability']:.4f}"

    )


    print()

    print(

        f"ENSEMBLE PROBABILITY    : "
        f"{result['ensemble_probability']:.4f}"

    )


    print(

        f"RISK LEVEL              : "
        f"{result['risk_level']}"

    )


    print()

    print(
        "=" * 60
    )


    print(
        " REAL-TIME PREDICTION COMPLETE"
    )


    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()