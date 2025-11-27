import pandas as pd
import numpy as np
import joblib
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
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)