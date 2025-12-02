"""
Example: Using the preprocessing pipeline in an API.
Simple, reusable patterns.
"""

from immo_eliza_ml import create_preprocessor, load_preprocessor, Predict
import pandas as pd


# ============================================================================
# TRAINING PHASE (run once to create models)
# ============================================================================

def train_and_save():
    """Example: Train models and save for API use."""
    # Load your data
    df = pd.read_csv("data/1_raw/properties.csv")

    # Create preprocessor with default config (includes capping)
    prep = create_preprocessor("default")

    # Or use custom config
    # prep = create_preprocessor({
    #     "features": ["living_area", "number_of_rooms", "region"],
    #     "use_capping": True,
    #     "capping_percentiles": (1, 99)
    # })

    # Fit and transform
    X_train, y_train = prep.fit_transform(df)

    # Save preprocessor as JSON (portable, no pickle!)
    prep.save("models/preprocessor.json")

    print("✓ Preprocessor saved to models/preprocessor.json")


# ============================================================================
# API USAGE (in your FastAPI/Flask app)
# ============================================================================

class PredictionAPI:
    """Simple API class. Initialize once, reuse many times."""

    def __init__(self):
        # Load preprocessor and models once at startup
        self.predictor = Predict(
            models_folder="models",
            preprocessor_path="models/preprocessor.json"
        )
        self.predictor.load()

    def predict_price(self, property_data: dict) -> float:
        """
        Predict price for a property.

        Args:
            property_data: Dict with property features
                {
                    "living_area": 120,
                    "number_of_rooms": 3,
                    "postal_code": 1000,
                    ...
                }

        Returns:
            Predicted price (float)
        """
        price = self.predictor.predict_single(
            property_data,
            model_name="XGBoost"  # or "Random Forest", etc.
        )
        return price


# ============================================================================
# FASTAPI EXAMPLE
# ============================================================================

"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
api = PredictionAPI()  # Initialize once


class PropertyRequest(BaseModel):
    living_area: float
    number_of_rooms: int
    postal_code: int
    # ... other fields


@app.post("/predict")
def predict_property_price(property: PropertyRequest):
    price = api.predict_price(property.dict())
    return {"predicted_price": price}
"""


# ============================================================================
# REUSABLE FOR OTHER ML TASKS
# ============================================================================

def custom_ml_task():
    """Example: Use preprocessing for a different ML task."""
    from immo_eliza_ml import FeaturePreprocessor

    # Define your own features
    my_features = ["living_area", "number_of_rooms", "garden_surface"]

    # Create preprocessor with custom config
    prep = FeaturePreprocessor(
        features=my_features,
        target="rental_price",  # Different target
        use_capping=True,
        capping_percentiles=(2, 98)  # Custom percentiles
    )

    # Use it like normal
    # X, y = prep.fit_transform(df)
    # ... train your model
    # prep.save("models/rental_preprocessor.json")


# ============================================================================
# USAGE PATTERNS
# ============================================================================

if __name__ == "__main__":
    print("Immo Eliza ML - API Usage Examples")
    print("=" * 50)

    print("\n1. Simple API initialization:")
    print("   api = PredictionAPI()")
    print("   price = api.predict_price(property_dict)")

    print("\n2. Preprocessor presets:")
    print("   default:         create_preprocessor('default')")
    print("   no_capping:      create_preprocessor('no_capping')")
    print("   strict_capping:  create_preprocessor('strict_capping')")

    print("\n3. Custom configuration:")
    print("   prep = create_preprocessor({")
    print("       'features': ['living_area', 'rooms'],")
    print("       'use_capping': True,")
    print("       'capping_percentiles': (1, 99)")
    print("   })")

    print("\n4. Load from saved:")
    print("   prep = load_preprocessor('models/preprocessor.json')")

    print("\n✓ All components are JSON-based (no pickle!)")
    print("✓ Simple, modular, reusable")
    print("✓ API-ready")
