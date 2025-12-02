"""
Predict class for Belgian real estate price prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from .preprocessing import FeaturePreprocessor


class Predict:
    """Makes price predictions for Belgian real estate."""

    # Default values for missing features
    FEATURE_DEFAULTS = {
        "postal_code": 1000,
        "living_area": 100,
        "number_of_rooms": 2,
        "number_of_facades": 2,
        "garden_surface": 0,
        "terrace_surface": 0,
        "equipped_kitchen": 0,
        "furnished": 0,
        "open_fire": 0,
        "terrace": 0,
        "garden": 0,
        "swimming_pool": 0,
        "state_of_building": "good",
        "subtype_of_property": "house",
    }

    def __init__(self, models_folder="models", preprocessor_path="models/preprocessor.json"):
        self.models = {}
        self.preprocessor = None
        self.models_folder = models_folder
        self.preprocessor_path = preprocessor_path
        self.default_model = "XGBoost"

    def _fill_missing_features(self, property_dict):
        """Fill missing features with defaults."""
        filled = self.FEATURE_DEFAULTS.copy()
        filled.update(property_dict)
        return filled

    def load(self):
        """Load preprocessor and all trained models."""
        # Load preprocessor
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = FeaturePreprocessor.load(self.preprocessor_path)
            print(f"Loaded preprocessor from {self.preprocessor_path}")
        else:
            print(f"Warning: Preprocessor not found at {self.preprocessor_path}")

        # Load models
        model_names = ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost"]
        for name in model_names:
            path = f"{self.models_folder}/{name.lower().replace(' ', '_')}.pkl"
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded model: {name}")

        print(f"\nLoaded {len(self.models)} models")
        return self

    def predict_single(self, property_dict, model_name=None):
        """
        Predict price for a single property.
        
        Args:
            property_dict: Dictionary with property features
            model_name: Model to use (default: XGBoost)
            
        Returns:
            Predicted price in euros
        """
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")

        # Fill missing features with defaults
        filled_dict = self._fill_missing_features(property_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame([filled_dict])
        
        # Transform features
        if self.preprocessor:
            X = self.preprocessor.transform(df)
        else:
            X = df
        
        # Predict (log price)
        log_price = self.models[model_name].predict(X)[0]
        
        # Convert back to actual price
        price = np.expm1(log_price)
        
        return price

    def predict_batch(self, properties_df, model_name=None):
        """
        Predict prices for multiple properties.
        
        Args:
            properties_df: DataFrame with property features
            model_name: Model to use (default: XGBoost)
            
        Returns:
            Array of predicted prices
        """
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")

        # Fill missing columns with defaults
        df = properties_df.copy()
        for col, default in self.FEATURE_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        # Transform features
        if self.preprocessor:
            X = self.preprocessor.transform(df)
        else:
            X = df
        
        # Predict and convert
        log_prices = self.models[model_name].predict(X)
        prices = np.expm1(log_prices)
        
        return prices

    def predict_all_models(self, property_dict):
        """
        Predict price using all available models.
        
        Returns:
            Dictionary with model names and predicted prices
        """
        results = {}
        
        # Fill missing features with defaults
        filled_dict = self._fill_missing_features(property_dict)
        
        df = pd.DataFrame([filled_dict])
        
        if self.preprocessor:
            X = self.preprocessor.transform(df)
        else:
            X = df
        
        for name, model in self.models.items():
            log_price = model.predict(X)[0]
            results[name] = np.expm1(log_price)
        
        return results

    def predict_with_confidence(self, property_dict):
        """
        Predict price with confidence range using all models.
        
        Returns:
            Dictionary with prediction, min, max, and std
        """
        predictions = self.predict_all_models(property_dict)
        prices = list(predictions.values())
        
        return {
            "predictions": predictions,
            "mean": np.mean(prices),
            "median": np.median(prices),
            "min": np.min(prices),
            "max": np.max(prices),
            "std": np.std(prices),
            "range": np.max(prices) - np.min(prices),
        }

    def display_prediction(self, property_dict, model_name=None):
        """Pretty print prediction for a property."""
        model_name = model_name or self.default_model
        price = self.predict_single(property_dict, model_name)
        
        print("=" * 50)
        print("🏠 PROPERTY DETAILS")
        print("=" * 50)
        
        for key, value in property_dict.items():
            label = key.replace("_", " ").title()
            if key in ["equipped_kitchen", "furnished", "open_fire", 
                       "terrace", "garden", "swimming_pool"]:
                value = "Yes" if value else "No"
            elif key in ["living_area", "garden_surface", "terrace_surface"]:
                value = f"{value} m²"
            print(f"  {label:<20}: {value}")
        
        print()
        print("=" * 50)
        print(f"💰 PREDICTED PRICE ({model_name}): €{price:,.0f}")
        print("=" * 50)
        
        return price

    def display_all_predictions(self, property_dict):
        """Show predictions from all models."""
        results = self.predict_with_confidence(property_dict)
        
        print("=" * 50)
        print("🏠 PROPERTY DETAILS")
        print("=" * 50)
        
        for key, value in property_dict.items():
            label = key.replace("_", " ").title()
            if key in ["equipped_kitchen", "furnished", "open_fire", 
                       "terrace", "garden", "swimming_pool"]:
                value = "Yes" if value else "No"
            elif key in ["living_area", "garden_surface", "terrace_surface"]:
                value = f"{value} m²"
            print(f"  {label:<20}: {value}")
        
        print()
        print("=" * 50)
        print("💰 PREDICTIONS BY MODEL")
        print("=" * 50)
        
        for name, price in results["predictions"].items():
            print(f"  {name:<20}: €{price:,.0f}")
        
        print("-" * 50)
        print(f"  {'Mean':<20}: €{results['mean']:,.0f}")
        print(f"  {'Median':<20}: €{results['median']:,.0f}")
        print(f"  {'Range':<20}: €{results['min']:,.0f} - €{results['max']:,.0f}")
        print("=" * 50)
        
        return results

    @staticmethod
    def sample_property():
        """Return a sample property dictionary."""
        return {
            "postal_code": 9000,
            "living_area": 120,
            "number_of_rooms": 3,
            "number_of_facades": 2,
            "equipped_kitchen": 1,
            "furnished": 0,
            "open_fire": 0,
            "terrace": 1,
            "terrace_surface": 15,
            "garden": 1,
            "garden_surface": 150,
            "swimming_pool": 0,
            "state_of_building": "good",
            "subtype_of_property": "house",
        }


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    # Demo without saved models (uses synthetic data)
    print("Demo mode (no saved models found)\n")
    
    # Create sample property
    property_data = {
        "postal_code": 9000,
        "living_area": 120,
        "number_of_rooms": 3,
        "number_of_facades": 2,
        "equipped_kitchen": 1,
        "swimming_pool": 0,
        "garden": 1,
        "garden_surface": 150,
    }
    
    print("Sample property:")
    for k, v in property_data.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 50)
    print("To use with your trained models:")
    print("=" * 50)
    print("""
    from predict import Predict
    
    # Load models
    predictor = Predict().load()
    
    # Predict single property
    price = predictor.predict_single(property_data)
    
    # Or with display
    predictor.display_prediction(property_data)
    
    # Compare all models
    predictor.display_all_predictions(property_data)
    """)