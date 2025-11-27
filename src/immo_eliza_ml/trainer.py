import numpy as np
import pandas as pd
import joblib
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
        """Save predictions for each model."""
        os.makedirs(folder, exist_ok=True)
        for name, preds in self.predictions.items():
            path = f"{folder}/{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(preds, path)
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
        """Load predictions for each model."""
        for name in self.models.keys():
            path = f"{folder}/{name.lower().replace(' ', '_')}.pkl"
            if os.path.exists(path):
                self.predictions[name] = joblib.load(path)
                print(f"Loaded {name} from {path}")
        print(f"\nAll predictions loaded from {folder}/")
        """Create a complete visual with all models and predictions."""
        os.makedirs(folder, exist_ok=True)
        n_models = len(self.predictions)
        
        # 1. Actual vs Predicted - All Models
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        fig.suptitle("Actual vs Predicted - All Models", fontsize=16, fontweight='bold')
        
        for i, (name, preds) in enumerate(self.predictions.items()):
            # Train
            axes[0, i].scatter(y_train, preds["train"], alpha=0.5, s=10, c='blue')
            axes[0, i].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
            axes[0, i].set_title(f"{name}\n(Train)")
            axes[0, i].set_xlabel("Actual")
            axes[0, i].set_ylabel("Predicted")
            
            # Test
            axes[1, i].scatter(y_test, preds["test"], alpha=0.5, s=10, c='green')
            axes[1, i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[1, i].set_title(f"{name}\n(Test)")
            axes[1, i].set_xlabel("Actual")
            axes[1, i].set_ylabel("Predicted")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/all_models_actual_vs_predicted.png", dpi=150)
        plt.close()
        print(f"Saved: {folder}/all_models_actual_vs_predicted.png")

        # 2. Residuals - All Models
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        fig.suptitle("Residuals - All Models", fontsize=16, fontweight='bold')
        
        for i, (name, preds) in enumerate(self.predictions.items()):
            train_res = y_train - preds["train"]
            test_res = y_test - preds["test"]
            
            # Train
            axes[0, i].scatter(preds["train"], train_res, alpha=0.5, s=10, c='blue')
            axes[0, i].axhline(y=0, color='r', linestyle='--')
            axes[0, i].set_title(f"{name}\n(Train)")
            axes[0, i].set_xlabel("Predicted")
            axes[0, i].set_ylabel("Residual")
            
            # Test
            axes[1, i].scatter(preds["test"], test_res, alpha=0.5, s=10, c='green')
            axes[1, i].axhline(y=0, color='r', linestyle='--')
            axes[1, i].set_title(f"{name}\n(Test)")
            axes[1, i].set_xlabel("Predicted")
            axes[1, i].set_ylabel("Residual")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/all_models_residuals.png", dpi=150)
        plt.close()
        print(f"Saved: {folder}/all_models_residuals.png")

        # 3. Error Distribution - All Models
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        fig.suptitle("Error Distribution - All Models", fontsize=16, fontweight='bold')
        
        for i, (name, preds) in enumerate(self.predictions.items()):
            train_res = y_train - preds["train"]
            test_res = y_test - preds["test"]
            
            # Train
            axes[0, i].hist(train_res, bins=30, edgecolor='black', alpha=0.7, color='blue')
            axes[0, i].axvline(x=0, color='r', linestyle='--')
            axes[0, i].set_title(f"{name}\n(Train) μ={train_res.mean():.3f}")
            axes[0, i].set_xlabel("Error")
            axes[0, i].set_ylabel("Frequency")
            
            # Test
            axes[1, i].hist(test_res, bins=30, edgecolor='black', alpha=0.7, color='green')
            axes[1, i].axvline(x=0, color='r', linestyle='--')
            axes[1, i].set_title(f"{name}\n(Test) μ={test_res.mean():.3f}")
            axes[1, i].set_xlabel("Error")
            axes[1, i].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(f"{folder}/all_models_error_distribution.png", dpi=150)
        plt.close()
        print(f"Saved: {folder}/all_models_error_distribution.png")

        # 4. Model Comparison Bar Chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Model Comparison", fontsize=16, fontweight='bold')
        
        names = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for name, preds in self.predictions.items():
            names.append(name)
            r2_scores.append(r2_score(y_test, preds["test"]))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds["test"])))
            mae_scores.append(mean_absolute_error(y_test, preds["test"]))
        
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

        print(f"\nAll plots saved to {folder}/")