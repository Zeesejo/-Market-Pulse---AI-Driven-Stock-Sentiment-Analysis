# Market Pulse - Prediction Function
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class MarketPulsePredictior:
    def __init__(self, model_dir="models"):
        """Load trained models and scalers"""
        self.cls_model = joblib.load(f"{model_dir}/best_classification_model.joblib")
        self.reg_model = joblib.load(f"{model_dir}/best_regression_model.joblib")
        self.cls_scaler = joblib.load(f"{model_dir}/classification_scaler.joblib")
        self.reg_scaler = joblib.load(f"{model_dir}/regression_scaler.joblib")
        self.feature_names = joblib.load(f"{model_dir}/feature_names.joblib")

        print(f"📈 Market Pulse Models Loaded")
        print(f"  Classification: XGBoost")
        print(f"  Regression: Random Forest")
        print(f"  Features: {len(self.feature_names)}")

    def predict_price_direction(self, features):
        """Predict if price will go up (1) or down (0)"""
        features_scaled = self.cls_scaler.transform(features)
        prediction = self.cls_model.predict(features_scaled)
        probability = self.cls_model.predict_proba(features_scaled)
        return prediction, probability

    def predict_future_return(self, features):
        """Predict future return percentage"""
        features_scaled = self.reg_scaler.transform(features)
        prediction = self.reg_model.predict(features_scaled)
        return prediction

    def predict_comprehensive(self, features_dict):
        """Make comprehensive predictions"""
        # Convert features dict to DataFrame with correct column order
        features_df = pd.DataFrame([features_dict])
        features_df = features_df.reindex(columns=self.feature_names, fill_value=0)

        # Make predictions
        direction, direction_proba = self.predict_price_direction(features_df)
        future_return = self.predict_future_return(features_df)

        return {
            'price_direction': 'UP' if direction[0] == 1 else 'DOWN',
            'direction_probability': max(direction_proba[0]),
            'predicted_return': future_return[0],
            'confidence': 'HIGH' if max(direction_proba[0]) > 0.7 else 'MEDIUM' if max(direction_proba[0]) > 0.6 else 'LOW'
        }

# Example usage:
# predictor = MarketPulsePredictior()
# prediction = predictor.predict_comprehensive(features_dict)
# print(f"Prediction: {prediction['price_direction']} with {prediction['confidence']} confidence")
