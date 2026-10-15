# ============================================================
# REAL-TIME HYBRID SOLAR FLARE PREDICTOR
#
# Pipeline:
#
# Current Solar Observation
#        |
#        +--> Tabular Features --> XGBoost
#        |
#        +--> Magnetogram --> ResNet18
#
# XGBoost Weight      = 0.2
# Magnetogram Weight  = 0.8
#
# Output:
# Hybrid flare probability + risk level
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
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).parent.parent


sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)


# ============================================================
# MODEL PATHS
# ============================================================

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
# BUILD RESNET18
# ============================================================

def build_magnetogram_model():

    print(
        "Building ResNet18 magnetogram model..."
    )


    model = models.resnet18(
        weights=None
    )


    # Magnetograms are grayscale

    model.conv1 = nn.Conv2d(

        in_channels=1,

        out_channels=64,

        kernel_size=7,

        stride=2,

        padding=3,

        bias=False

    )


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
# REAL-TIME HYBRID PREDICTOR
# ============================================================

class RealTimeSolarFlarePredictor:


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
            " LOADING REAL-TIME HYBRID SYSTEM"
        )


        print(
            "=" * 60
        )


        print(
            f"Using device: {self.device}"
        )


        # ====================================================
        # LOAD XGBOOST
        # ====================================================

        print(
            "\nLoading XGBoost..."
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
            package["model"]
        )


        self.imputer = (
            package["imputer"]
        )


        self.features = (
            package["features"]
        )


        print(
            "XGBoost loaded successfully."
        )


        # ====================================================
        # LOAD MAGNETOGRAM MODEL
        # ====================================================

        print(
            "\nLoading magnetogram ResNet18..."
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
            " REAL-TIME HYBRID SYSTEM READY"
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


        # Convert Series to DataFrame

        if isinstance(
            feature_row,
            pd.Series
        ):

            feature_row = (

                feature_row
                .to_frame()
                .T

            )


        # Check features

        missing_features = [

            feature

            for feature in self.features

            if feature not in feature_row.columns

        ]


        if missing_features:

            raise ValueError(

                "Missing required XGBoost features:\n"

                f"{missing_features}"

            )


        # Exact feature order

        X = (

            feature_row[
                self.features
            ]

            .copy()

        )


        # Apply fitted imputer

        X = (

            self.imputer.transform(
                X
            )

        )


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
        # Accept path
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

                    f"Magnetogram not found:\n"
                    f"{magnetogram_path}"

                )


            # ------------------------------------------------
            # NPZ FILE
            # ------------------------------------------------

            if (
                magnetogram_path.suffix
                == ".npz"
            ):

                magnetogram = np.load(

                    magnetogram_path

                )[
                    "img"
                ]


            else:

                raise ValueError(

                    "Currently supported magnetogram "
                    "format is .npz"

                )


        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        magnetogram = np.asarray(
            magnetogram
        )


        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        if magnetogram.ndim != 2:

            raise ValueError(

                "Magnetogram must be 2D.\n"

                f"Received shape: "

                f"{magnetogram.shape}"

            )


        # ----------------------------------------------------
        # Convert:
        #
        # [H, W]
        #
        # to:
        #
        # [1, 1, H, W]
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
        # CNN Prediction
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
    # FULL REAL-TIME PREDICTION
    # ========================================================

    def predict(

        self,

        realtime_features,

        realtime_magnetogram

    ):


        print()

        print(
            "=" * 60
        )


        print(
            " GENERATING REAL-TIME HYBRID PREDICTION"
        )


        print(
            "=" * 60
        )


        # XGBoost

        xgb_probability = (

            self.predict_xgb(

                realtime_features

            )

        )


        # Magnetogram

        magnetogram_probability = (

            self.predict_magnetogram(

                realtime_magnetogram

            )

        )


        # Ensemble

        ensemble_probability = (

            XGB_WEIGHT
            * xgb_probability

            +

            MAGNETOGRAM_WEIGHT
            * magnetogram_probability

        )


        risk_level = (

            self.classify_risk(

                ensemble_probability

            )

        )


        result = {

            "xgb_probability":

                float(
                    xgb_probability
                ),

            "magnetogram_probability":

                float(
                    magnetogram_probability
                ),

            "ensemble_probability":

                float(
                    ensemble_probability
                ),

            "risk_level":

                risk_level

        }


        return result