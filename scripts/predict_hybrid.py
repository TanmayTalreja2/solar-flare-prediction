# ============================================================
# HYBRID REAL-TIME SOLAR FLARE PREDICTOR
#
# Uses:
#   1. XGBoost tabular feature model
#   2. ResNet18 magnetogram model
#   3. Hybrid ensemble
#   4. Human-readable report generator
#
# Ensemble:
#   20% XGBoost
#   80% Magnetogram ResNet18
# ============================================================


import sys

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import torch
import torch.nn as nn

import torchvision.models as models
ENSEMBLE_THRESHOLD = 0.5


# ============================================================
# PROJECT PATH
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
# PATHS
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
# ENSEMBLE WEIGHTS
# ============================================================

XGB_WEIGHT = 0.2

MAGNETOGRAM_WEIGHT = 0.8


# ============================================================
# BUILD MAGNETOGRAM RESNET18
# ============================================================

def build_magnetogram_model():

    print(
        "Building ResNet18 magnetogram model..."
    )


    model = models.resnet18(
        weights=None
    )


    # --------------------------------------------------------
    # Magnetograms are grayscale
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
# HYBRID PREDICTOR
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


        # ----------------------------------------------------
        # LOAD XGBOOST
        # ----------------------------------------------------

        print(
            "\nLoading XGBoost model..."
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


        # ----------------------------------------------------
        # LOAD MAGNETOGRAM MODEL
        # ----------------------------------------------------

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
        # HANDLE CHECKPOINT FORMAT
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
        # REMOVE DATAPARALLEL PREFIX
        # ----------------------------------------------------

        cleaned_state_dict = {}


        for key, value in (
            state_dict.items()
        ):


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
        # LOAD WEIGHTS
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
        # Convert Series to DataFrame if needed
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
        # Imputation
        # ----------------------------------------------------

        X = (

            self.imputer.transform(
                X
            )

        )


        # ----------------------------------------------------
        # Probability
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
        # numpy array
        #
        # OR
        #
        # path to .npz file
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
        # Convert to numpy
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
        # Convert to tensor
        #
        # Original:
        # [H, W]
        #
        # CNN expects:
        # [Batch, Channel, H, W]
        # ----------------------------------------------------

        tensor = torch.tensor(

            magnetogram,

            dtype=torch.float32

        ).unsqueeze(
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


        # ----------------------------------------------------
        # Simple human-readable risk categories
        #
        # These are operational categories.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        result = {

            "xgb_probability":
                xgb_probability,

            "magnetogram_probability":
                magnetogram_probability,

            "ensemble_probability":
                ensemble_probability,

            "risk_level":
                risk_level

        }


        return result


# ============================================================
# QUICK TEST
# ============================================================

def main():


    print()

    print(
        "=" * 60
    )


    print(
        " HYBRID SOLAR FLARE PREDICTOR TEST"
    )


    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Load predictor
    # --------------------------------------------------------

    predictor = (

        HybridSolarFlarePredictor()

    )


    # --------------------------------------------------------
    # Historical data
    #
    # This section is ONLY for testing the realtime predictor.
    #
    # Later, realtime data will replace this.
    # --------------------------------------------------------

    data_path = (

        PROJECT_ROOT
        / "data"
        / "processed"
        / "features"
        / "sharp_goes_temporal_features_2012_full.parquet"

    )


    if not data_path.exists():

        raise FileNotFoundError(

            f"Data file not found:\n"
            f"{data_path}"

        )


    print(
        "\nLoading test observation..."
    )


    df = pd.read_parquet(
        data_path
    )


    df[
        "observation_time"
    ] = pd.to_datetime(

        df[
            "observation_time"
        ]

    )


# --------------------------------------------------------
# Select a test observation that ACTUALLY has
# a corresponding magnetogram file.
# --------------------------------------------------------

    test_df = (

        df[
            df[
            "observation_time"
        ]

        > pd.Timestamp(
            "2012-06-30 23:59:59"
        )

    ]

    .copy()

)


    print(
    "\nSearching for a test observation "
    "with an available magnetogram..."
)


    input_row = None

    magnetogram_path = None


    for _, row in test_df.iterrows():


    # ----------------------------------------------------
    # IMPORTANT:
    #
    # For inference we don't know target_24h.
    #
    # But for this historical test, the magnetogram files
    # were saved with t0 or t1.
    #
    # We try both filenames.
    # ----------------------------------------------------

        base_filename = (

        f"harp_{row['HARPNUM']}_"

        f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}"

    )


        possible_paths = [

        PROJECT_ROOT
        / "data"
        / "processed"
        / "magnetograms"
        / f"{base_filename}_t0.npz",


        PROJECT_ROOT
        / "data"
        / "processed"
        / "magnetograms"
        / f"{base_filename}_t1.npz"

    ]


        for path in possible_paths:


            if path.exists():


                input_row = row

                magnetogram_path = path

                break


            if input_row is not None:

                break


# --------------------------------------------------------
# Safety check
# --------------------------------------------------------

    if input_row is None:


        raise RuntimeError(

        "Could not find any test observation "
        "with a matching magnetogram."

    )


# --------------------------------------------------------
# Display selected observation
# --------------------------------------------------------

    print()

    print(
    f"Observation time: "
    f"{input_row['observation_time']}"
)


    print(
    f"HARPNUM: "
    f"{input_row['HARPNUM']}"
)


    print(
    f"Magnetogram: "
    f"{magnetogram_path.name}"
)

    # --------------------------------------------------------
    # Build magnetogram filename
    # --------------------------------------------------------

    magnetogram_filename = (

        f"harp_{input_row['HARPNUM']}_"

        f"{input_row['observation_time'].strftime('%Y%m%d_%H%M%S')}_"

        f"t{int(input_row['target_24h'])}.npz"

    )


    magnetogram_path = (

        PROJECT_ROOT
        / "data"
        / "processed"
        / "magnetograms"
        / magnetogram_filename

    )


    print()

    print(
        f"Observation time: "
        f"{input_row['observation_time']}"
    )


    print(
        f"HARPNUM: "
        f"{input_row['HARPNUM']}"
    )


    print(
        f"Magnetogram: "
        f"{magnetogram_path.name}"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = (

        predictor.predict(

            feature_row=input_row,

            magnetogram=magnetogram_path

        )

    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()

    print(
        "=" * 60
    )


    print(
        " HYBRID PREDICTION"
    )


    print(
        "=" * 60
    )


    print()

    print(

        f"XGBoost Probability      : "
        f"{result['xgb_probability']:.4f}"

    )


    print(

        f"Magnetogram Probability  : "
        f"{result['magnetogram_probability']:.4f}"

    )


    print()

    print(

        f"ENSEMBLE PROBABILITY     : "
        f"{result['ensemble_probability']:.4f}"

    )


    print(

        f"RISK LEVEL               : "
        f"{result['risk_level']}"

    )


    print()

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()