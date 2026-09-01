import sys
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------
# Add src and project root to Python path
# ------------------------------------------------------------

project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# ------------------------------------------------------------
# Import project modules
# ------------------------------------------------------------

from scripts.predict import SolarFlarePredictor
from solarflare.reporting.report_generator import (
    SolarFlareReportGenerator
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

data_path = (
    project_root
    / "data"
    / "processed"
    / "features"
    / "sharp_goes_temporal_features_2012_full.parquet"
)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("Loading prediction model...")

    predictor = SolarFlarePredictor(
        model_path=str(
            project_root
            / "models"
            / "xgboost_2012_temporal_features.joblib"
        )
    )

    print("Loading processed data...")

    if not data_path.exists():

        print(
            f"ERROR: Data file not found:\n{data_path}"
        )

        return

    df = pd.read_parquet(data_path)

    # --------------------------------------------------------
    # Use the test period
    # --------------------------------------------------------

    df["observation_time"] = pd.to_datetime(
        df["observation_time"]
    )

    test_df = df[
        df["observation_time"]
        > pd.Timestamp("2012-06-30 23:59:59")
    ].copy()

    if len(test_df) == 0:

        print("ERROR: No test observations found.")

        return

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        f"Generating predictions for "
        f"{len(test_df)} observations..."
    )

    predictions = predictor.predict(
        test_df
    )

    # --------------------------------------------------------
    # Select highest-risk observation
    # --------------------------------------------------------

    best_index = predictions[
        "flare_probability"
    ].idxmax()

    input_row = test_df.loc[
        best_index
    ]

    prediction_row = predictions.loc[
        best_index
    ]

    print()
    print(
        "Highest-risk observation selected:"
    )

    print(
        f"Probability: "
        f"{prediction_row['flare_probability']:.4f}"
    )

    print(
        f"Risk: "
        f"{prediction_row['risk_level']}"
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    generator = SolarFlareReportGenerator(
        output_dir=project_root / "results" / "reports"
    )

    report_path = generator.generate_report(
        input_row=input_row,
        prediction_row=prediction_row,
        model=predictor.model
    )

    print()
    print("=" * 55)
    print("REPORT GENERATED SUCCESSFULLY")
    print("=" * 55)
    print()
    print(
        f"Report saved to:\n{report_path}"
    )


if __name__ == "__main__":
    main()