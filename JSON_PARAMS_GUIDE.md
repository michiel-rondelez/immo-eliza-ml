# JSON Model Parameters Guide

## Overview

The `ModelTrainer` class now supports saving and loading model **parameters** (hyperparameters) as JSON files. This is useful for:

1. **Documentation**: Track model configurations in human-readable format
2. **Version Control**: Store model configs in git (unlike binary .pkl files)
3. **Reproducibility**: Recreate model architecture for retraining
4. **Transparency**: Easily review and compare model hyperparameters

## Important Distinction

### JSON Parameters vs PKL Models

- **JSON files** (`.json`): Store **only hyperparameters** (model configuration)
  - Contains model type and initialization parameters
  - Human-readable and git-friendly
  - Used to recreate model architecture
  - **Does NOT contain trained weights**

- **PKL files** (`.pkl`): Store **complete fitted models**
  - Contains model type, parameters, AND trained weights
  - Binary format (not human-readable)
  - Used to make predictions without retraining
  - **Required for using trained models**

## Usage

### Saving Model Parameters to JSON

```python
from src.immo_eliza_ml.trainer import ModelTrainer

# Create and train models
trainer = ModelTrainer()
trainer.train(X_train, X_test, y_train, y_test)

# Save model parameters as JSON
trainer.save_model_params_to_json(folder="models")
```

This creates JSON files like:
- `models/linear_regression_params.json`
- `models/random_forest_params.json`
- `models/xgboost_params.json`
- etc.

### Loading Model Parameters from JSON

```python
from src.immo_eliza_ml.trainer import ModelTrainer

# Create new trainer
trainer = ModelTrainer()

# Load model configurations from JSON
trainer.load_model_params_from_json(folder="models")

# Models are recreated but NOT trained
# You need to train them or load fitted .pkl files
trainer.train(X_train, X_test, y_train, y_test)  # Option 1: Retrain
# OR
trainer.load_training_models("models")  # Option 2: Load fitted models
```

## Example JSON Output

Here's what a Random Forest model parameters JSON file looks like:

```json
{
  "model_type": "RandomForestRegressor",
  "model_name": "Random Forest",
  "parameters": {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "bootstrap": true,
    "criterion": "squared_error",
    "n_jobs": null,
    "verbose": 0
  }
}
```

## Complete Workflow

### 1. Training and Saving

```python
from src.immo_eliza_ml.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train(X_train, X_test, y_train, y_test)

# Save BOTH formats:
trainer.save_training_models("models")              # .pkl files (fitted models)
trainer.save_model_params_to_json("models")         # .json files (parameters)
```

### 2. Loading for Predictions

```python
from src.immo_eliza_ml.trainer import ModelTrainer

trainer = ModelTrainer()

# Load fitted models (for predictions)
trainer.load_training_models("models")  # Loads .pkl files

# Make predictions
predictions = trainer.models["Random Forest"].predict(X_new)
```

### 3. Loading for Retraining

```python
from src.immo_eliza_ml.trainer import ModelTrainer

trainer = ModelTrainer()

# Load model architecture from JSON
trainer.load_model_params_from_json("models")  # Loads .json files

# Retrain with new data
trainer.train(X_train_new, X_test_new, y_train_new, y_test_new)
```

## Use Cases

### 1. Version Control
Store JSON files in git to track model configuration changes over time:

```bash
git add models/*.json
git commit -m "Update Random Forest max_depth from 20 to 10"
```

### 2. Model Comparison
Easily compare hyperparameters across different model versions:

```bash
diff models_v1/random_forest_params.json models_v2/random_forest_params.json
```

### 3. Documentation
Include JSON files in documentation to show exact model configurations used.

### 4. Reproducibility
Share JSON files with team members to ensure everyone uses the same model configuration.

## Benefits Over Saving Self

Previously, if you saved `self` (the entire `ModelTrainer` object), you would:
- ❌ Save unnecessary data (predictions, results, etc.)
- ❌ Create large binary files
- ❌ Make it hard to inspect model configurations
- ❌ Make version control difficult

With JSON parameters, you:
- ✅ Save only what's needed (model type and hyperparameters)
- ✅ Create small, readable files
- ✅ Make it easy to inspect and compare configurations
- ✅ Enable version control of model configs

## Methods

### `save_model_params_to_json(folder="models")`

Saves model parameters to JSON files.

**Parameters:**
- `folder` (str): Directory to save JSON files (default: "models")

**Creates:**
- One JSON file per model with naming pattern: `{model_name}_params.json`

### `load_model_params_from_json(folder="models")`

Loads model parameters from JSON files and recreates model architecture.

**Parameters:**
- `folder` (str): Directory containing JSON files (default: "models")

**Returns:**
- `dict`: Dictionary of recreated models (not fitted)

**Note:** Models need to be trained or fitted .pkl files need to be loaded before making predictions.
