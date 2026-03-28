"""
Immo Eliza ML - Simple, reusable ML pipeline for real estate prediction.
API-ready components.
"""

# Core preprocessing
from .preprocessing import FeaturePreprocessor, OutlierCapper

# Simple config helpers for API use
from .config import create_preprocessor, load_preprocessor, PreprocessorConfig

# Model training
from .trainer import ModelTrainer

# Predictions
from .predict import Predict

__all__ = [
    # Preprocessing
    "FeaturePreprocessor",
    "OutlierCapper",
    # Config helpers (simple API usage)
    "create_preprocessor",
    "load_preprocessor",
    "PreprocessorConfig",
    # Training
    "ModelTrainer",
    # Prediction
    "Predict",
]
