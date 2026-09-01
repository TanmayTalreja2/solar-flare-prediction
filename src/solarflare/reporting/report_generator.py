from pathlib import Path
from datetime import datetime


class SolarFlareReportGenerator:
    """
    Generates a human-readable solar flare prediction report.
    """

    def __init__(self, output_dir="results/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def format_value(value):
        """Format numerical values for human readability."""

        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)

        if abs(value) >= 1e6 or (0 < abs(value) < 1e-3):
            return f"{value:.3e}"

        return f"{value:.3f}"

    @staticmethod
    def readable_name(name):
        """Convert technical feature names to readable names."""

        replacements = {
            "USFLUX": "Total Unsigned Magnetic Flux",
            "TOTUSJH": "Total Current Helicity",
            "TOTUSJZ": "Total Vertical Current",
            "TOTUSJH_CHANGE_6H": "Total Current Helicity Change (6h)",
            "TOTUSJZ_CHANGE_6H": "Total Vertical Current Change (6h)",
            "MEANGAM": "Mean Magnetic Field Angle",
            "MEANGBT": "Mean Gradient of Total Magnetic Field",
            "MEANJZH": "Mean Current Helicity",
            "MEANJZD": "Mean Vertical Current Density",
            "R_VALUE": "Magnetic R-Value",
            "SHRGT45": "Fraction of Area with Shear >45°",
            "TOTPOT": "Total Magnetic Free Energy Proxy",
            "TOTUSJH_CHANGE_6H": "6-Hour Helicity Change",
        }

        if name in replacements:
            return replacements[name]

        return (
            str(name)
            .replace("_", " ")
            .title()
        )

    def generate_report(
        self,
        input_row,
        prediction_row,
        model=None,
        filename=None
    ):
        probability = float(
            prediction_row["flare_probability"]
        )

        risk_level = prediction_row["risk_level"]
        prediction = int(prediction_row["prediction"])

        # --------------------------------------------------
        # Observation information
        # --------------------------------------------------

        if "NOAA_AR" in prediction_row:
            active_region = prediction_row["NOAA_AR"]
        elif "NOAA_AR" in input_row.index:
            active_region = input_row["NOAA_AR"]
        else:
            active_region = "Not available"

        if "observation_time" in prediction_row:
            observation_time = prediction_row["observation_time"]
        elif "observation_time" in input_row.index:
            observation_time = input_row["observation_time"]
        else:
            observation_time = "Not available"

        if hasattr(observation_time, "strftime"):
            observation_time = observation_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # --------------------------------------------------
        # Interpretation
        # --------------------------------------------------

        if risk_level == "LOW RISK":
            interpretation = (
                "The model predicts a relatively low likelihood "
                "of an M/X-class solar flare within the next "
                "24 hours."
            )

        elif risk_level == "MODERATE RISK":
            interpretation = (
                "The model indicates a moderate likelihood "
                "of an M/X-class solar flare within the next "
                "24 hours. Continued monitoring is recommended."
            )

        elif risk_level == "HIGH RISK":
            interpretation = (
                "The model indicates an elevated likelihood "
                "of an M/X-class solar flare within the next "
                "24 hours. The active region should be monitored "
                "closely."
            )

        else:
            interpretation = (
                "The model indicates a very high predicted risk "
                "of an M/X-class solar flare within the next "
                "24 hours. The active region warrants close "
                "monitoring."
            )

        # --------------------------------------------------
        # Model-important features
        # --------------------------------------------------

        important_features = []

        # Identifiers should NEVER be shown as model indicators.
        excluded_features = {
            "HARPNUM",
            "NOAA_AR",
            "observation_time",
            "target_24h",
        }

        if model is not None and hasattr(
            model,
            "feature_importances_"
        ):

            importances = model.feature_importances_

            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            else:
                feature_names = input_row.index

            ranked = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )

            for feature_name, importance in ranked:

                if feature_name in excluded_features:
                    continue

                if feature_name not in input_row.index:
                    continue

                value = input_row[feature_name]

                # Skip missing values
                if value is None:
                    continue

                try:
                    if value != value:  # NaN
                        continue
                except Exception:
                    pass

                important_features.append(
                    (
                        feature_name,
                        value,
                        float(importance)
                    )
                )

                if len(important_features) == 5:
                    break

        # --------------------------------------------------
        # Build report
        # --------------------------------------------------

        lines = []

        lines.append("# ☀️ Solar Flare Prediction Report")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Observation
        lines.append("## 🔭 Observation")
        lines.append("")
        lines.append(
            f"**Active Region:** {active_region}"
        )
        lines.append(
            f"**Observation Time:** {observation_time}"
        )
        lines.append(
            "**Prediction Window:** Next 24 Hours"
        )
        lines.append("")

        # Prediction
        lines.append("## 🚨 Model Prediction")
        lines.append("")

        lines.append(
            f"### **{probability * 100:.2f}% predicted probability**"
        )
        lines.append("")

        lines.append(
            f"**Risk Level:** **{risk_level}**"
        )

        if prediction == 1:
            lines.append(
                "**Model Decision:** FLARE RISK DETECTED"
            )
        else:
            lines.append(
                "**Model Decision:** NO FLARE RISK DETECTED"
            )

        lines.append("")

        # Interpretation
        lines.append("## 🧠 Interpretation")
        lines.append("")
        lines.append(interpretation)
        lines.append("")

        lines.append(
            "The reported probability is the probability "
            "predicted by the trained XGBoost model; it should "
            "not be interpreted as a guaranteed physical "
            "probability of a flare occurring."
        )

        lines.append("")

        # Features
        if important_features:

            lines.append("## 🧲 Key Magnetic Indicators")
            lines.append("")

            lines.append(
                "The following features had the highest "
                "global importance among the model inputs "
                "for this prediction system:"
            )

            lines.append("")

            for feature_name, value, importance in important_features:

                lines.append(
                    f"- **{self.readable_name(feature_name)}**  "
                    f"  Value: `{self.format_value(value)}`  "
                    f"  Model importance: `{importance:.4f}`"
                )

            lines.append("")

        # Method
        lines.append("## ⚙️ Method")
        lines.append("")

        lines.append(
            "The prediction was generated using an XGBoost "
            "model trained on 71 magnetic and temporal features "
            "derived from SHARP observations."
        )

        lines.append("")

        lines.append(
            "Missing values were handled using the median "
            "imputer fitted on the training data."
        )

        lines.append("")

        lines.append(
            "> **Important:** Model feature importance indicates "
            "which inputs were influential to the trained model "
            "overall. It does not imply a causal relationship."
        )

        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append(
            f"Report generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        report_text = "\n".join(lines)

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        if filename is None:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                f"solar_flare_report_{timestamp}.md"
            )

        output_path = self.output_dir / filename

        output_path.write_text(
            report_text,
            encoding="utf-8"
        )

        return output_path