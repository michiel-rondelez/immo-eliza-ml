"""
FastAPI REST API for Immo Eliza ML
Deploy this as a web service for programmatic access.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from immo_eliza_ml.predict import Predict
import os

# Initialize FastAPI app
app = FastAPI(
    title="Immo Eliza ML API",
    description="REST API for Belgian real estate price prediction",
    version="1.0.0"
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor once at startup
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load models when the API starts."""
    global predictor
    try:
        predictor = Predict(
            models_folder="models",
            preprocessor_path="models/preprocessor.json"
        )
        predictor.load()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("Make sure models are trained first!")

# Request/Response models
class PropertyRequest(BaseModel):
    """Property data for prediction."""
    postal_code: int = Field(..., ge=1000, le=9999, description="Belgian postal code")
    living_area: float = Field(..., gt=0, description="Living area in m²")
    number_of_rooms: int = Field(..., gt=0, description="Number of rooms")
    number_of_facades: int = Field(..., ge=1, description="Number of facades")
    equipped_kitchen: int = Field(0, ge=0, le=1, description="Has equipped kitchen (0/1)")
    furnished: int = Field(0, ge=0, le=1, description="Is furnished (0/1)")
    open_fire: int = Field(0, ge=0, le=1, description="Has open fire (0/1)")
    terrace: int = Field(0, ge=0, le=1, description="Has terrace (0/1)")
    garden: int = Field(0, ge=0, le=1, description="Has garden (0/1)")
    swimming_pool: int = Field(0, ge=0, le=1, description="Has swimming pool (0/1)")
    garden_surface: float = Field(0.0, ge=0, description="Garden surface in m²")
    terrace_surface: float = Field(0.0, ge=0, description="Terrace surface in m²")
    state_of_building: str = Field("good", description="State: good, as_new, to_renovate, etc.")
    subtype_of_property: str = Field("house", description="Type: house, apartment, villa, etc.")

class PredictionResponse(BaseModel):
    """Single prediction response."""
    predicted_price: float
    model: str
    property_data: Dict

class AllPredictionsResponse(BaseModel):
    """All models prediction response."""
    predictions: Dict[str, float]
    statistics: Dict[str, float]

# API Endpoints
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Immo Eliza ML API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict/all": "POST - Predictions from all models",
            "/models": "GET - List available models",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy",
        "models_loaded": len(predictor.models) if predictor else 0
    }

@app.get("/models")
async def list_models():
    """List all available models."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "models": list(predictor.models.keys()),
        "default": predictor.default_model
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(
    property_data: PropertyRequest,
    model: Optional[str] = None
):
    """
    Predict property price using a single model.
    
    - **model**: Model name (default: XGBoost). Options: Linear Regression, Decision Tree, Random Forest, SVR, XGBoost
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")
    
    model_name = model or predictor.default_model
    
    if model_name not in predictor.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: {list(predictor.models.keys())}"
        )
    
    try:
        price = predictor.predict_single(
            property_data.dict(),
            model_name=model_name
        )
        
        return PredictionResponse(
            predicted_price=price,
            model=model_name,
            property_data=property_data.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/all", response_model=AllPredictionsResponse)
async def predict_all_models(property_data: PropertyRequest):
    """
    Get predictions from all available models with statistics.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = predictor.predict_all_models(property_data.dict())
        stats = predictor.predict_with_confidence(property_data.dict())
        
        return AllPredictionsResponse(
            predictions=predictions,
            statistics=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

