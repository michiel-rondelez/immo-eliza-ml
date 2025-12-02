import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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
    Auto-detects feature types and applies correct pipeline.
    
    Usage:
        # Select features - pipeline auto-assigned
        prep = FeaturePreprocessor(
            features=["living_area", "region", "swimming_pool"],
            target="price"
        )
        
        # All features (default)
        prep = FeaturePreprocessor()
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

    def __init__(self, features=None, target="price", log_target=True):
        self.features = set(features) if features else self.ALL_FEATURES
        self.target = target
        self.log_target = log_target
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
        """Build pipeline based on detected feature types."""
        transformers = []

        if self.numeric:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), self.numeric))

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
        """Show which features go to which pipeline."""
        print(f"Numeric ({len(self.numeric)}):     {self.numeric}")
        print(f"Categorical ({len(self.categorical)}): {self.categorical}")
        print(f"Binary ({len(self.binary)}):      {self.binary}")
        print(f"Target:            {self.target} (log={self.log_target})")

    def save(self, path):
        """Save preprocessor configuration and fitted parameters as JSON."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")

        config = {
            "features": list(self.features),
            "target": self.target,
            "log_target": self.log_target,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "binary": self.binary,
            "fitted_params": {}
        }

        # Extract fitted parameters from pipeline
        for name, transformer, columns in self.pipeline.transformers_:
            if name == "num":
                # Extract StandardScaler parameters
                imputer = transformer.named_steps["impute"]
                scaler = transformer.named_steps["scale"]
                config["fitted_params"]["numeric"] = {
                    "imputer_statistics": imputer.statistics_.tolist(),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist()
                }
            elif name == "cat":
                # Extract OneHotEncoder parameters
                imputer = transformer.named_steps["impute"]
                encoder = transformer.named_steps["encode"]
                config["fitted_params"]["categorical"] = {
                    "imputer_fill_value": imputer.statistics_[0] if hasattr(imputer, "statistics_") else "Unknown",
                    "encoder_categories": [cat.tolist() for cat in encoder.categories_]
                }
            elif name == "bin":
                # Extract binary imputer parameters
                config["fitted_params"]["binary"] = {
                    "imputer_statistics": transformer.statistics_.tolist()
                }

        # Save to JSON
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load preprocessor from JSON and reconstruct pipeline."""
        with open(path, 'r') as f:
            config = json.load(f)

        # Create instance with saved configuration
        instance = cls(
            features=config["features"],
            target=config["target"],
            log_target=config["log_target"]
        )

        # Rebuild pipeline
        transformers = []
        fitted_params = config["fitted_params"]

        if instance.numeric:
            # Reconstruct numeric pipeline with fitted parameters
            imputer = SimpleImputer(strategy="median")
            imputer.statistics_ = np.array(fitted_params["numeric"]["imputer_statistics"])

            scaler = StandardScaler()
            scaler.mean_ = np.array(fitted_params["numeric"]["scaler_mean"])
            scaler.scale_ = np.array(fitted_params["numeric"]["scaler_scale"])
            scaler.n_features_in_ = len(scaler.mean_)

            transformers.append(("num", Pipeline([
                ("impute", imputer),
                ("scale", scaler),
            ]), instance.numeric))

        if instance.categorical:
            # Reconstruct categorical pipeline with fitted parameters
            imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
            if "imputer_fill_value" in fitted_params["categorical"]:
                imputer.statistics_ = np.array([fitted_params["categorical"]["imputer_fill_value"]])

            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.categories_ = [np.array(cat) for cat in fitted_params["categorical"]["encoder_categories"]]
            encoder.n_features_in_ = len(encoder.categories_)

            transformers.append(("cat", Pipeline([
                ("impute", imputer),
                ("encode", encoder),
            ]), instance.categorical))

        if instance.binary:
            # Reconstruct binary imputer with fitted parameters
            imputer = SimpleImputer(strategy="constant", fill_value=0)
            imputer.statistics_ = np.array(fitted_params["binary"]["imputer_statistics"])

            transformers.append(("bin", imputer, instance.binary))

        instance.pipeline = ColumnTransformer(transformers)
        return instance