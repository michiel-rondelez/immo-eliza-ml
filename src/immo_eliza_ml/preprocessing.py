import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers to percentile limits. Simple and reusable."""

    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """Learn percentile bounds from data."""
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        """Cap values to learned bounds."""
        X_capped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_capped

    def get_params_dict(self):
        """Get parameters as dict for JSON serialization."""
        return {
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_bounds": self.lower_bounds_.tolist() if self.lower_bounds_ is not None else None,
            "upper_bounds": self.upper_bounds_.tolist() if self.upper_bounds_ is not None else None
        }

    def set_params_dict(self, params):
        """Set parameters from dict (for loading from JSON)."""
        self.lower_percentile = params["lower_percentile"]
        self.upper_percentile = params["upper_percentile"]
        self.lower_bounds_ = np.array(params["lower_bounds"]) if params["lower_bounds"] else None
        self.upper_bounds_ = np.array(params["upper_bounds"]) if params["upper_bounds"] else None
        return self


def get_region(postal_code):
    try:
        pc = int(postal_code)
        regions = {
            (1000, 1299): 'Brussels', (1300, 1499): 'Walloon_Brabant',
            (1500, 1999): 'Flemish_Brabant', (2000, 2999): 'Antwerp',
            (3000, 3499): 'Flemish_Brabant', (3500, 3999): 'Limburg',
            (4000, 4999): 'Liege', (5000, 5999): 'Namur',
            (6000, 6599): 'Hainaut', (6600, 6999): 'Luxembourg',
            (7000, 7999): 'Hainaut', (8000, 8999): 'West_Flanders',
            (9000, 9999): 'East_Flanders',
        }
        for (low, high), region in regions.items():
            if low <= pc <= high:
                return region
    except:
        pass
    return "Unknown"


class FeaturePreprocessor:
    """
    Simple, reusable preprocessing pipeline. API-ready.

    Usage:
        # Basic usage
        prep = FeaturePreprocessor()

        # Custom features with capping
        prep = FeaturePreprocessor(
            features=["living_area", "region", "swimming_pool"],
            target="price",
            use_capping=True,
            capping_percentiles=(1, 99)
        )
    """

    # Feature type definitions
    NUMERIC = {
        "number_of_rooms", "living_area", "number_of_facades",
        "garden_surface", "terrace_surface", "postal_code",
        "total_outdoor", "outdoor_ratio", "luxury_score",
        "area_log", "area_per_room",
    }

    CATEGORICAL = {"subtype_of_property", "state_of_building", "region"}

    BINARY = {"equipped_kitchen", "furnished", "open_fire",
              "terrace", "garden", "swimming_pool"}

    ALL_FEATURES = NUMERIC | CATEGORICAL | BINARY

    def __init__(self, features=None, target="price", log_target=True,
                 use_capping=True, capping_percentiles=(1, 99)):
        """
        Initialize preprocessor.

        Args:
            features: List of features to use (default: all)
            target: Target variable name
            log_target: Apply log transform to target
            use_capping: Cap outliers in numeric features
            capping_percentiles: (lower, upper) percentiles for capping
        """
        self.features = set(features) if features else self.ALL_FEATURES
        self.target = target
        self.log_target = log_target
        self.use_capping = use_capping
        self.capping_percentiles = capping_percentiles
        self.pipeline = None

        # Auto-assign features to correct type
        self.numeric = list(self.features & self.NUMERIC)
        self.categorical = list(self.features & self.CATEGORICAL)
        self.binary = list(self.features & self.BINARY)

    def _engineer(self, df):
        """Add engineered features."""
        df = df.copy()

        df["region"] = df["postal_code"].apply(get_region)
        
        df["total_outdoor"] = (
            df["garden_surface"].fillna(0) + df["terrace_surface"].fillna(0)
        )
        df["outdoor_ratio"] = df["total_outdoor"] / (df["living_area"] + 1)
        
        df["area_log"] = np.log1p(df["living_area"])
        df["area_per_room"] = df["living_area"] / df["number_of_rooms"].replace(0, 1)
        
        df["luxury_score"] = (
            df["equipped_kitchen"].fillna(0) +
            df["furnished"].fillna(0) +
            df["open_fire"].fillna(0) +
            df["swimming_pool"].fillna(0) * 2
        )

        return df

    def _build_pipeline(self):
        """Build pipeline based on config. Simple and modular."""
        transformers = []

        if self.numeric:
            # Build numeric pipeline steps
            steps = [("impute", SimpleImputer(strategy="median"))]

            # Add capping if enabled
            if self.use_capping:
                steps.append(("cap", OutlierCapper(
                    lower_percentile=self.capping_percentiles[0],
                    upper_percentile=self.capping_percentiles[1]
                )))

            # Always scale last
            steps.append(("scale", StandardScaler()))

            transformers.append(("num", Pipeline(steps), self.numeric))

        if self.categorical:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.categorical))

        if self.binary:
            transformers.append(("bin",
                SimpleImputer(strategy="constant", fill_value=0),
                self.binary))

        return ColumnTransformer(transformers)

    def fit_transform(self, df):
        """Fit and transform training data."""
        X = self._engineer(df)
        y = np.log1p(df[self.target]) if self.log_target else df[self.target]

        self.pipeline = self._build_pipeline()
        return self.pipeline.fit_transform(X), y

    def transform(self, df):
        """Transform new data."""
        X = self._engineer(df)
        return self.pipeline.transform(X)

    def get_target(self, df):
        """Get target variable with same transform as fit_transform."""
        return np.log1p(df[self.target]) if self.log_target else df[self.target]

    @property
    def feature_names(self):
        return list(self.pipeline.get_feature_names_out())

    def info(self):
        """Show pipeline configuration."""
        print(f"Numeric ({len(self.numeric)}):     {self.numeric}")
        print(f"Categorical ({len(self.categorical)}): {self.categorical}")
        print(f"Binary ({len(self.binary)}):      {self.binary}")
        print(f"Target:            {self.target} (log={self.log_target})")
        print(f"Outlier Capping:   {self.use_capping}")
        if self.use_capping:
            print(f"  Percentiles:     {self.capping_percentiles}")

    def save(self, path):
        """
        Save preprocessor config to JSON **and** the fitted pipeline to pickle.

        - JSON: configuration (features, target, capping settings)
        - PKL:  fitted `ColumnTransformer` pipeline (used at prediction time)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")

        # Derive matching .json / .pkl paths
        if path.endswith(".json"):
            json_path = path
            pkl_path = path.replace(".json", ".pkl")
        elif path.endswith(".pkl"):
            pkl_path = path
            json_path = path.replace(".pkl", ".json")
        else:
            json_path = f"{path}.json"
            pkl_path = f"{path}.pkl"

        # Lightweight, portable config
        config = {
            "features": list(self.features),
            "target": self.target,
            "log_target": self.log_target,
            "use_capping": self.use_capping,
            "capping_percentiles": list(self.capping_percentiles),
            "numeric": self.numeric,
            "categorical": self.categorical,
            "binary": self.binary,
        }

        # Save fitted pipeline
        joblib.dump(self.pipeline, pkl_path)

        # Save config as JSON
        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved preprocessor pipeline to {pkl_path}")
        print(f"Saved preprocessor metadata to {json_path}")

    @classmethod
    def load(cls, path):
        """
        Load preprocessor:
        - Reads config from JSON
        - Loads fitted pipeline from the matching .pkl file
        """
        # Work out matching paths
        if path.endswith(".json"):
            json_path = path
            pkl_path = path.replace(".json", ".pkl")
        elif path.endswith(".pkl"):
            pkl_path = path
            json_path = path.replace(".pkl", ".json")
        else:
            json_path = f"{path}.json"
            pkl_path = f"{path}.pkl"

        # Load config (if available)
        try:
            with open(json_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {
                "features": list(cls.ALL_FEATURES),
                "target": "price",
                "log_target": True,
                "use_capping": False,
                "capping_percentiles": [1, 99],
            }

        # Recreate instance with same configuration
        instance = cls(
            features=config.get("features", list(cls.ALL_FEATURES)),
            target=config.get("target", "price"),
            log_target=config.get("log_target", True),
            use_capping=config.get("use_capping", False),
            capping_percentiles=tuple(config.get("capping_percentiles", (1, 99))),
        )

        # Load fitted pipeline
        instance.pipeline = joblib.load(pkl_path)

        return instance