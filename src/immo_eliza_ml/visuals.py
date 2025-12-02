# predictions.py

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Visualizations:
    """Loads and visualizes stored model predictions."""

    def __init__(self):
        self.predictions = {}
        self.models = {}
        self.results = None
        self.y_train = None
        self.y_test = None

    def load_all(self, models_folder="models", predictions_folder="predictions", data_folder="data"):
        """Load models, predictions, and y values."""
        
        # Load y values
        y_path = f"{data_folder}/y_values.json"
        if os.path.exists(y_path):
            with open(y_path, 'r') as f:
                y_data = json.load(f)
            self.y_train = np.array(y_data["y_train"])
            self.y_test = np.array(y_data["y_test"])
            print(f"Loaded y values from {y_path}")
        else:
            print(f"Warning: {y_path} not found")

        # Load models
        model_names = ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost"]
        for name in model_names:
            path = f"{models_folder}/{name.lower().replace(' ', '_')}.pkl"
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded model: {name}")

        # Load results
        results_path = f"{models_folder}/results.csv"
        if os.path.exists(results_path):
            self.results = pd.read_csv(results_path)
            print(f"Loaded results from {results_path}")

        # Load predictions (pickled dicts with 'train' and 'test' arrays)
        for name in model_names:
            path = f"{predictions_folder}/{name.lower().replace(' ', '_')}.pkl"
            if os.path.exists(path):
                self.predictions[name] = joblib.load(path)
                print(f"Loaded predictions: {name}")

        print("\nAll data loaded!")

    def summary(self):
        """Print results summary."""
        if self.results is None:
            print("No results loaded")
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

    # ========== ITERATION METHODS ==========

    def iterate_predictions(self, dataset="test"):
        """Iterate over all predictions."""
        for name, preds in self.predictions.items():
            yield name, preds[dataset]

    def iterate_with_actual(self, dataset="test"):
        """Iterate over predictions with actual values."""
        y_actual = self.y_test if dataset == "test" else self.y_train
        for name, preds in self.predictions.items():
            yield name, preds[dataset], y_actual

    def iterate_with_errors(self, dataset="test"):
        """Iterate over predictions with errors."""
        y_actual = self.y_test if dataset == "test" else self.y_train
        for name, preds in self.predictions.items():
            errors = y_actual - preds[dataset]
            yield name, preds[dataset], y_actual, errors

    # ========== VISUALIZATION METHODS ==========

    def plot_all(self, folder="plots"):
        """Create all visualizations."""
        os.makedirs(folder, exist_ok=True)
        
        self.plot_actual_vs_predicted(folder)
        self.plot_residuals(folder)
        self.plot_error_distribution(folder)
        self.plot_model_comparison(folder)
        self.plot_individual_models(folder)
        
        print(f"\nAll plots saved to {folder}/")

    def plot_actual_vs_predicted(self, folder="plots"):
        """Plot actual vs predicted for all models."""
        os.makedirs(folder, exist_ok=True)
        n_models = len(self.predictions)
        
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        fig.suptitle("Actual vs Predicted - All Models", fontsize=16, fontweight='bold')
        
        for i, (name, preds, y_actual, errors) in enumerate(self.iterate_with_errors("train")):
            axes[0, i].scatter(self.y_train, preds, alpha=0.5, s=10, c='blue')
            axes[0, i].plot([self.y_train.min(), self.y_train.max()], 
                           [self.y_train.min(), self.y_train.max()], 'r--')
            axes[0, i].set_title(f"{name}\n(Train)")
            axes[0, i].set_xlabel("Actual")
            axes[0, i].set_ylabel("Predicted")
        
        for i, (name, preds, y_actual, errors) in enumerate(self.iterate_with_errors("test")):
            axes[1, i].scatter(self.y_test, preds, alpha=0.5, s=10, c='green')
            axes[1, i].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--')
            axes[1, i].set_title(f"{name}\n(Test)")
            axes[1, i].set_xlabel("Actual")
            axes[1, i].set_ylabel("Predicted")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/all_models_actual_vs_predicted.png", dpi=150)
        plt.close()
        print(f"Saved: {folder}/all_models_actual_vs_predicted.png")

    def plot_residuals(self, folder="plots"):
        """Plot residuals for all models."""
        os.makedirs(folder, exist_ok=True)
        n_models = len(self.predictions)
        
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        fig.suptitle("Residuals - All Models", fontsize=16, fontweight='bold')
        
        for i, (name, preds, y_actual, errors) in enumerate(self.iterate_with_errors("train")):
            axes[0, i].scatter(preds, errors, alpha=0.5, s=10, c='blue')
            axes[0, i].axhline(y=0, color='r', linestyle='--')
            axes[0, i].set_title(f"{name}\n(Train)")
            axes[0, i].set_xlabel("Predicted")
            axes[0, i].set_ylabel("Residual")
        
        for i, (name, preds, y_actual, errors) in enumerate(self.iterate_with_errors("test")):
            axes[1, i].scatter(preds, errors, alpha=0.5, s=10, c='green')
            axes[1, i].axhline(y=0, color='r', linestyle='--')
            axes[1, i].set_title(f"{name}\n(Test)")
            axes[1, i].set_xlabel("Predicted")
            axes[1, i].set_ylabel("Residual")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/all_models_residuals.png", dpi=150)
        plt.close()
        print(f"Saved: {folder}/all_models_residuals.png")

    def plot_error_distribution(self, folder="plots"):
        """Plot error distribution for all models."""
        os.makedirs(folder, exist_ok=True)
        n_models = len(self.predictions)
        
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        fig.suptitle("Error Distribution - All Models", fontsize=16, fontweight='bold')
        
        for i, (name, preds, y_actual, errors) in enumerate(self.iterate_with_errors("train")):
            axes[0, i].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='blue')
            axes[0, i].axvline(x=0, color='r', linestyle='--')
            axes[0, i].set_title(f"{name}\n(Train) μ={errors.mean():.3f}")
            axes[0, i].set_xlabel("Error")
            axes[0, i].set_ylabel("Frequency")
        
        for i, (name, preds, y_actual, errors) in enumerate(self.iterate_with_errors("test")):
            axes[1, i].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='green')
            axes[1, i].axvline(x=0, color='r', linestyle='--')
            axes[1, i].set_title(f"{name}\n(Test) μ={errors.mean():.3f}")
            axes[1, i].set_xlabel("Error")
            axes[1, i].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/all_models_error_distribution.png", dpi=150)
        plt.close()

        print(f"Saved: {folder}/all_models_error_distribution.png")
    
    def plot_model_comparison(self, folder="plots"):
        """Bar chart comparing model performance."""
        os.makedirs(folder, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Model Comparison", fontsize=16, fontweight='bold')
        
        names = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for name, preds, y_actual, errors in self.iterate_with_errors("test"):
            names.append(name)
            r2_scores.append(r2_score(y_actual, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_actual, preds)))
            mae_scores.append(mean_absolute_error(y_actual, preds))
        
        colors = ['steelblue', 'coral', 'seagreen', 'gold', 'purple']
        
        axes[0].bar(names, r2_scores, color=colors)
        axes[0].set_title("R² Score (higher is better)")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(min(r2_scores) - 0.05, max(r2_scores) + 0.05)
        
        axes[1].bar(names, rmse_scores, color=colors)
        axes[1].set_title("RMSE (lower is better)")
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(names, mae_scores, color=colors)
        axes[2].set_title("MAE (lower is better)")
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{folder}/model_comparison.png", dpi=150)
        plt.close()
        print(f"Saved: {folder}/model_comparison.png")

    def plot_individual_models(self, folder="plots"):
        """Create individual plots for each model."""
        os.makedirs(folder, exist_ok=True)
        
        for name, preds, y_actual, errors in self.iterate_with_errors("test"):
            filename = name.lower().replace(" ", "_")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{name}", fontsize=16, fontweight='bold')
            
            # Actual vs Predicted
            axes[0].scatter(y_actual, preds, alpha=0.5, s=10)
            axes[0].plot([y_actual.min(), y_actual.max()], 
                        [y_actual.min(), y_actual.max()], 'r--')
            axes[0].set_title("Actual vs Predicted")
            axes[0].set_xlabel("Actual")
            axes[0].set_ylabel("Predicted")
            
            # Residuals
            axes[1].scatter(preds, errors, alpha=0.5, s=10)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_title("Residuals")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Residual")
            
            # Error Distribution
            axes[2].hist(errors, bins=30, edgecolor='black', alpha=0.7)
            axes[2].axvline(x=0, color='r', linestyle='--')
            axes[2].set_title(f"Error Distribution (μ={errors.mean():.3f})")
            axes[2].set_xlabel("Error")
            axes[2].set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.savefig(f"{folder}/{filename}_analysis.png", dpi=150)
            plt.close()
            print(f"Saved: {folder}/{filename}_analysis.png")

    
    def get_best_model(self):
        """Get the best performing model."""
        best_name = None
        best_r2 = -np.inf
        
        for name, preds, y_actual, errors in self.iterate_with_errors("test"):
            r2 = r2_score(y_actual, preds)
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
        
        return best_name, best_r2

    def predictions_to_dataframe(self, dataset="test"):
        """Get all predictions as DataFrame."""
        data = {}
        for name, preds in self.iterate_predictions(dataset):
            data[name] = preds
        return pd.DataFrame(data)

    def find_bad_predictions(self, model_name="XGBoost", threshold=1.0):
        """Find predictions with error above threshold."""
        preds = self.predictions[model_name]["test"]
        errors = np.abs(self.y_test - preds)
        bad_indices = np.where(errors > threshold)[0]
        
        return pd.DataFrame({
            "index": bad_indices,
            "actual": self.y_test.iloc[bad_indices] if hasattr(self.y_test, 'iloc') else self.y_test[bad_indices],
            "predicted": preds[bad_indices],
            "error": errors[bad_indices]
        })
