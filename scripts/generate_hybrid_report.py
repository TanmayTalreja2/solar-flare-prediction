# ============================================================
# HYBRID SOLAR FLARE REPORT GENERATOR
#
# Uses:
#
#   1. HybridSolarFlarePredictor
#       - XGBoost tabular model
#       - ResNet18 magnetogram model
#       - 20% / 80% ensemble
#
#   2. SolarFlareReportGenerator
#       - Human-readable report
#
# ============================================================


import sys

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).parent.parent


# ============================================================
# PYTHON PATH
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from scripts.predict_hybrid import (
    ENSEMBLE_THRESHOLD,
    HybridSolarFlarePredictor,
    XGB_WEIGHT,
    MAGNETOGRAM_WEIGHT
)


from solarflare.reporting.report_generator import (
    SolarFlareReportGenerator
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


REPORTS_DIR = (

    PROJECT_ROOT
    / "results"
    / "reports"

)


# ============================================================
# TEST PERIOD
# ============================================================

TEST_START = pd.Timestamp(
    "2012-07-01 00:00:00"
)


# ============================================================
# FIND OBSERVATION WITH MAGNETOGRAM
# ============================================================

def find_valid_observation(
    df
):


    print()

    print(
        "Searching for an observation "
        "with an available magnetogram..."
    )


    for _, row in df.iterrows():


        # ----------------------------------------------------
        # Build filename base
        # ----------------------------------------------------

        base_filename = (

            f"harp_{row['HARPNUM']}_"

            f"{row['observation_time'].strftime('%Y%m%d_%H%M%S')}"

        )


        # ----------------------------------------------------
        # IMPORTANT
        #
        # During inference we do not know target_24h.
        #
        # Historical magnetogram files contain either:
        #
        # t0
        # OR
        # t1
        #
        # Therefore try both.
        # ----------------------------------------------------

        possible_paths = [

            MAGNETOGRAM_DIR
            / f"{base_filename}_t0.npz",

            MAGNETOGRAM_DIR
            / f"{base_filename}_t1.npz"

        ]


        for path in possible_paths:


            if path.exists():


                return (

                    row,

                    path

                )


    return (

        None,

        None

    )


# ============================================================
# MAIN
# ============================================================

def main():


    print()

    print(
        "=" * 60
    )


    print(
        " HYBRID SOLAR FLARE REPORT GENERATION"
    )


    print(
        "=" * 60
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    REPORTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # LOAD HYBRID PREDICTOR
    # ========================================================

    print()

    print(
        "Initializing hybrid prediction system..."
    )


    predictor = (

        HybridSolarFlarePredictor()

    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    print()

    print(
        "Loading processed solar observations..."
    )


    if not DATA_PATH.exists():


        raise FileNotFoundError(

            f"Processed data not found:\n"
            f"{DATA_PATH}"

        )


    df = pd.read_parquet(
        DATA_PATH
    )


    # ========================================================
    # CONVERT TIME
    # ========================================================

    df[
        "observation_time"
    ] = pd.to_datetime(

        df[
            "observation_time"
        ]

    )


    # ========================================================
    # TEST PERIOD
    # ========================================================

    test_df = (

        df[

            df[
                "observation_time"
            ]

            >= TEST_START

        ]

        .copy()

    )


    print()

    print(

        f"Total observations: "
        f"{len(df)}"

    )


    print(

        f"Test-period observations: "
        f"{len(test_df)}"

    )


    if len(
        test_df
    ) == 0:


        raise RuntimeError(

            "No test-period observations found."

        )


    # ========================================================
    # FIND VALID OBSERVATION
    # ========================================================

    input_row, magnetogram_path = (

        find_valid_observation(
            test_df
        )

    )


    if input_row is None:


        raise RuntimeError(

            "Could not find an observation "
            "with an available magnetogram."

        )


    # ========================================================
    # DISPLAY SELECTED OBSERVATION
    # ========================================================

    print()

    print(
        "=" * 60
    )


    print(
        " SELECTED SOLAR OBSERVATION"
    )


    print(
        "=" * 60
    )


    print()

    print(

        f"HARPNUM: "
        f"{input_row['HARPNUM']}"

    )


    if (

        "NOAA_AR"
        in input_row.index

    ):


        print(

            f"NOAA AR: "
            f"{input_row['NOAA_AR']}"

        )


    print(

        f"Observation time: "
        f"{input_row['observation_time']}"

    )


    print(

        f"Magnetogram file: "
        f"{magnetogram_path.name}"

    )


    # ========================================================
    # GENERATE HYBRID PREDICTION
    # ========================================================

    print()

    print(
        "=" * 60
    )


    print(
        " GENERATING HYBRID PREDICTION"
    )


    print(
        "=" * 60
    )


    result = (

        predictor.predict(

            feature_row=input_row,

            magnetogram=magnetogram_path

        )

    )


    # ========================================================
    # DISPLAY MODEL RESULTS
    # ========================================================

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

        f"Ensemble Probability     : "
        f"{result['ensemble_probability']:.4f}"

    )


    print(

        f"Risk Level               : "
        f"{result['risk_level']}"

    )


    # ========================================================
    # BUILD REPORT PREDICTION ROW
    #
    # Compatible with the existing
    # SolarFlareReportGenerator.
    # ========================================================

    prediction_value = int(
    result["ensemble_probability"]
    >= ENSEMBLE_THRESHOLD
)


    prediction_row = pd.Series({

        # ----------------------------------------------------
        # Required existing report generator fields
        # ----------------------------------------------------

        "flare_probability":

            result[
                "ensemble_probability"
            ],


        "prediction":

            prediction_value,


        "risk_level":

            result[
                "risk_level"
            ],


        # ----------------------------------------------------
        # Hybrid information
        # ----------------------------------------------------

        "xgb_probability":

            result[
                "xgb_probability"
            ],


        "magnetogram_probability":

            result[
                "magnetogram_probability"
            ],


        "ensemble_probability":

            result[
                "ensemble_probability"
            ],


        "xgb_weight":

            XGB_WEIGHT,


        "magnetogram_weight":

            MAGNETOGRAM_WEIGHT

    })


    # ========================================================
    # GENERATE REPORT
    # ========================================================

    print()

    print(
        "=" * 60
    )


    print(
        " GENERATING HUMAN-READABLE REPORT"
    )


    print(
        "=" * 60
    )


    generator = (

        SolarFlareReportGenerator(

            output_dir=REPORTS_DIR

        )

    )


    report_path = (

        generator.generate_report(

            input_row=input_row,

            prediction_row=prediction_row,

            # The existing report generator
            # uses the XGBoost model for
            # feature importance explanation.

            model=predictor.xgb_model

        )

    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "=" * 60
    )


    print(
        " HYBRID REPORT GENERATED SUCCESSFULLY"
    )


    print(
        "=" * 60
    )


    print()

    print(
        "Hybrid Ensemble:"
    )


    print(

        f"XGBoost Weight      : "
        f"{XGB_WEIGHT:.1f}"

    )


    print(

        f"Magnetogram Weight  : "
        f"{MAGNETOGRAM_WEIGHT:.1f}"

    )


    print()

    print(

        f"Final Probability   : "
        f"{result['ensemble_probability']:.4f}"

    )


    print(

        f"Risk Level          : "
        f"{result['risk_level']}"

    )


    print()

    print(

        f"Report saved to:\n"
        f"{report_path}"

    )


    print()

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()