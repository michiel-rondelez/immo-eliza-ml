import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    """
    Trains:
    - Linear Regression (baseline)
    - Random Forest (baseline)
    - Decision Tree (baseline)
    - XGBoost (if installed)
    
    Then performs GridSearchCV and stores
    the best variant of each model type.
    """

    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.best_model_name = None

    # ---------------------------
    # Baseline model training
    # ---------------------------
    def train_baseline_models(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}

        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results["Linear Regression"] = r2_score(y_test, lr.predict(X_test))
        self.models["linear_regression"] = lr

        # 2. Random Forest
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=25,
            random_state=42
        )
        rf.fit(X_train, y_train)
        results["Random Forest"] = r2_score(y_test, rf.predict(X_test))
        self.models["random_forest"] = rf

        # 3. Decision Tree
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        results["Decision Tree"] = r2_score(y_test, dt.predict(X_test))
        self.models["decision_tree"] = dt

        # 4. Optional XGBoost
        if XGBOOST_AVAILABLE:
            xgb = XGBRegressor()
            xgb.fit(X_train, y_train)
            results["XGBoost"] = r2_score(y_test, xgb.predict(X_test))
            self.models["xgboost"] = xgb

        return results

    # ---------------------------
    # Grid Search & CV
    # ---------------------------
    def grid_search_all(self, X, y):

        config = {
            "linear_regression": {
                "model": LinearRegression(),
                "params": {}
            },
            "lasso": {
                "model": Lasso(max_iter=5000),
                "params": {"alpha": [0.1, 0.5, 1.0, 2.0]}
            },
            "decision_tree": {
                "model": DecisionTreeRegressor(),
                "params": {
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 10, 30],
                },
            },
            "random_forest": {
                "model": RandomForestRegressor(),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                },
            },
        }

        if XGBOOST_AVAILABLE:
            config["xgboost"] = {
                "model": XGBRegressor(),
                "params": {
                    "n_estimators": [200, 400],
                    "max_depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                },
            }

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        results = []

        for name, cfg in config.items():

            gs = GridSearchCV(
                cfg["model"],
                cfg["params"],
                cv=cv,
                scoring="r2",
                verbose=0
            )

            gs.fit(X, y)

            results.append({
                "model": name,
                "best_score": gs.best_score_,
                "best_params": gs.best_params_
            })

            self.best_models[name] = gs.best_estimator_

        results_df = pd.DataFrame(results)
        self.best_model_name = results_df.iloc[results_df["best_score"].idxmax()]["model"]

        return results_df

    # ---------------------------
    # Save best models
    # ---------------------------
    def save_models(self, folder):
        for name, model in self.best_models.items():
            joblib.dump(model, f"{folder}/{name}_best.pkl")

        print(f"💾 Saved best model versions to: {folder}")
