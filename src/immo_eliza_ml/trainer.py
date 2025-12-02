import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class ModelTrainer:
    """Trains and compares regression models."""

    def __init__(self):
        self.predictions = {}
        self.models = {
            "Linear Regression": LinearRegression(),
            "SVR": SVR(kernel="rbf", C=1.0),
             # Decision Tree - reduce max_depth
            "Decision Tree": DecisionTreeRegressor(
                max_depth=8,           # Was 20, reduce it
                min_samples_split=20,   # Require more samples to split
                min_samples_leaf=10,     # Require more samples per leaf
                random_state=42
            ),

            # Random Forest - reduce depth, add min_samples
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,           # Was 20, reduce it
                min_samples_split=10,   # Minimum samples to split a node
                min_samples_leaf=5,     # Minimum samples per leaf
                max_features="sqrt",    # Use sqrt of features per tree
                random_state=42
            ),

            # XGBoost - reduce complexity, add regularization
            "XGBoost": XGBRegressor(
                n_estimators=200,
                max_depth=4,            # Was 6, reduce it
                learning_rate=0.05,     # Was 0.1, slower learning
                reg_alpha=0.1,          # L1 regularization (Lasso)
                reg_lambda=1.0,         # L2 regularization (Ridge)
                subsample=0.8,          # Use 80% of data per tree
                colsample_bytree=0.8,   # Use 80% of features per tree
                random_state=42,
                verbosity=0
            ),
        }
        self.results = None

    def train(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate."""
        results = []
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            self.predictions[name] = {"train": train_pred, "test": test_pred}
            results.append({
                "model": name,
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "test_mae": mean_absolute_error(y_test, test_pred),
            })
        self.results = pd.DataFrame(results)
        return self.results

    def get_predictions(self):
        """Get stored predictions."""
        return self.predictions

    def summary(self):
        """Print results summary."""
        if self.results is None:
            return
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        baseline = self.results[self.results["model"] == "Linear Regression"].iloc[0]
        best = self.results.loc[self.results["test_r2"].idxmax()]
        print(f"\nBaseline (Linear Regression): R² = {baseline['test_r2']:.4f}")
        print(f"Best ({best['model']}): R² = {best['test_r2']:.4f}")
        print(f"Difference: +{best['test_r2'] - baseline['test_r2']:.4f}")
        print("\n" + "-" * 60)
        print(self.results[["model", "train_r2", "test_r2", "test_rmse", "test_mae"]].to_string(index=False))
        print("-" * 60)

    def overfitting_summary(self):
        """Print quick overfitting summary."""
        print("\n" + "=" * 50)
        print("OVERFITTING SUMMARY")
        print("=" * 50)
        print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Gap':>10} {'Status':>15}")
        print("-" * 65)
        
        for _, row in self.results.iterrows():
            gap = row["train_r2"] - row["test_r2"]
            
            if gap > 0.10:
                status = "OVERFIT ⚠️"
            elif gap > 0.05:
                status = "MODERATE ⚡"
            else:
                status = "OK ✅"
            
            print(f"{row['model']:<20} {row['train_r2']:>10.4f} {row['test_r2']:>10.4f} {gap:>10.4f} {status:>15}")
        
        print("-" * 65)

    # ========== SAVE METHODS ==========

    def save_training_models(self, folder="models"):
        """Save models and results."""
        os.makedirs(folder, exist_ok=True)
        for name, model in self.models.items():
            path = f"{folder}/{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, path)
        if self.results is not None:
            self.results.to_csv(f"{folder}/results.csv", index=False)
        print(f"Saved models to {folder}/")

    def save_predictions_models(self, folder="predictions"):
        """Save predictions for each model as JSON."""
        os.makedirs(folder, exist_ok=True)
        for name, preds in self.predictions.items():
            path = f"{folder}/{name.lower().replace(' ', '_')}.json"
            # Convert numpy array to list for JSON serialization
            predictions_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
            with open(path, 'w') as f:
                json.dump(predictions_list, f, indent=2)
            print(f"Saved {name} to {path}")
        print(f"\nAll predictions saved to {folder}/")

    # ========== LOAD METHODS ==========

    def load_training_models(self, folder="models"):
        """Load models and results."""
        for name in self.models.keys():
            path = f"{folder}/{name.lower().replace(' ', '_')}.pkl"
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded {name} from {path}")
        
        results_path = f"{folder}/results.csv"
        if os.path.exists(results_path):
            self.results = pd.read_csv(results_path)
            print(f"Loaded results from {results_path}")
        
        print(f"\nAll models loaded from {folder}/")

    def load_predictions_models(self, folder="predictions"):
        """Load predictions for each model from JSON."""
        for name in self.models.keys():
            path = f"{folder}/{name.lower().replace(' ', '_')}.json"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.predictions[name] = np.array(json.load(f))
                print(f"Loaded {name} from {path}")
        print(f"\nAll predictions loaded from {folder}/")

    # ========== JSON SAVE/LOAD METHODS ==========

    def save_model_params_to_json(self, folder="models"):
        """Save model parameters (not the fitted model) to JSON files.

        This saves only the model type and hyperparameters, not the trained weights.
        Useful for documenting model configurations and recreating model architectures.
        """
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            # Get model parameters
            params = model.get_params()

            # Convert numpy types to Python native types for JSON serialization
            serializable_params = {}
            for key, value in params.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_params[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_params[key] = value.tolist()
                elif value is None or isinstance(value, (int, float, str, bool)):
                    serializable_params[key] = value
                else:
                    # For complex objects, store as string
                    serializable_params[key] = str(value)

            # Create model config with type and parameters
            model_config = {
                "model_type": type(model).__name__,
                "model_name": name,
                "parameters": serializable_params
            }

            # Save to JSON
            json_path = f"{folder}/{name.lower().replace(' ', '_')}_params.json"
            with open(json_path, 'w') as f:
                json.dump(model_config, f, indent=2)

            print(f"Saved {name} parameters to {json_path}")

        print(f"\nAll model parameters saved to {folder}/")

    def load_model_params_from_json(self, folder="models"):
        """Load model parameters from JSON files and recreate models.

        Note: This recreates the model architecture but does NOT restore trained weights.
        To use trained models, you still need to load the .pkl files using load_training_models().
        """
        # Mapping of model type names to classes
        model_classes = {
            "LinearRegression": LinearRegression,
            "SVR": SVR,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "XGBRegressor": XGBRegressor,
        }

        loaded_models = {}

        for name in list(self.models.keys()):
            json_path = f"{folder}/{name.lower().replace(' ', '_')}_params.json"

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    model_config = json.load(f)

                model_type = model_config["model_type"]
                params = model_config["parameters"]

                # Get the model class
                if model_type in model_classes:
                    model_class = model_classes[model_type]

                    # Create new model instance with loaded parameters
                    loaded_models[name] = model_class(**params)
                    print(f"Loaded {name} parameters from {json_path}")
                else:
                    print(f"Warning: Unknown model type '{model_type}' for {name}")
            else:
                print(f"Warning: File not found: {json_path}")

        if loaded_models:
            self.models.update(loaded_models)
            print(f"\nModel architectures recreated from {folder}/")
            print("Note: Models need to be retrained or load fitted .pkl files to make predictions")

        return loaded_models
