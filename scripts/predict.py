import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class SolarFlarePredictor:
    """
    Inference module for the Solar Flare Prediction project.
    Uses the trained XGBoost model and pre-fitted median imputer
    to generate probabilities and risk categories.
    """
    
    def __init__(self, model_path="models/xgboost_2012_temporal_features.joblib"):
        """
        Initialize the predictor by loading the saved model package.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        print(f"Loading model from {self.model_path}...")
        self.model_package = joblib.load(self.model_path)
        
        self.model = self.model_package["model"]
        self.imputer = self.model_package["imputer"]
        self.features = self.model_package["features"]
        
        # We override the default threshold (0.5) with the optimal tuned threshold for F1/Recall.
        self.threshold = 0.01
        print("Model loaded successfully.")
        
    def classify_risk(self, probability):
        """
        Map a flare probability to a risk category based on project requirements.
        """
        if probability < 0.01:
            return "LOW RISK"
        elif 0.01 <= probability < 0.05:
            return "MODERATE RISK"
        elif 0.05 <= probability < 0.20:
            return "HIGH RISK"
        else:
            return "VERY HIGH RISK"
            
    def predict(self, df):
        """
        Make predictions on a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing at least the 71 required features.
            
        Returns:
            pd.DataFrame: A dataframe containing original indices/keys, 
                          flare_probability, risk_level, and prediction (0/1).
        """
        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Extract only the features the model was trained on, in the exact order
        X = df[self.features].copy()
        
        # Apply median imputation (fitted on the training set)
        X_imputed = self.imputer.transform(X)
        
        # Generate probabilities for the positive class (flare)
        probabilities = self.model.predict_proba(X_imputed)[:, 1]
        
        # Generate predictions (1 if prob >= threshold else 0)
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Create risk levels
        risk_levels = [self.classify_risk(p) for p in probabilities]
        
        # Build results DataFrame
        results = pd.DataFrame({
            "flare_probability": probabilities,
            "prediction": predictions,
            "risk_level": risk_levels
        }, index=df.index)
        
        # Try to include identifying information if present
        if "NOAA_AR" in df.columns:
            results.insert(0, "NOAA_AR", df["NOAA_AR"])
        if "observation_time" in df.columns:
            results.insert(1, "observation_time", df["observation_time"])
            
        return results

if __name__ == "__main__":
    # Test the predictor on a small sample of the processed test data
    try:
        predictor = SolarFlarePredictor()
        
        data_path = Path("data/processed/features/sharp_goes_temporal_features_2012_full.parquet")
        if data_path.exists():
            print(f"Loading a sample from {data_path} to test predictor...")
            # We'll just load the first 100 rows for a quick test
            df = pd.read_parquet(data_path).tail(100)  # tail because test data is at the end (July-Dec)
            
            results = predictor.predict(df)
            print("\nPrediction Results Sample:")
            print(results.head())
            
            print("\nRisk Level Distribution:")
            print(results["risk_level"].value_counts())
    except Exception as e:
        print(f"Error during quick test: {e}")
