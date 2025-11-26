import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def get_region(postal_code):
    try:
        pc = int(postal_code)
        ranges = [
            ((1000, 1299), 'Brussels'),
            ((1300, 1499), 'Walloon_Brabant'),
            ((1500, 1999), 'Flemish_Brabant'),
            ((2000, 2999), 'Antwerp'),
            ((3000, 3499), 'Flemish_Brabant'),
            ((3500, 3999), 'Limburg'),
            ((4000, 4999), 'Liege'),
            ((5000, 5999), 'Namur'),
            ((6000, 6599), 'Hainaut'),
            ((6600, 6999), 'Luxembourg'),
            ((7000, 7999), 'Hainaut'),
            ((8000, 8999), 'West_Flanders'),
            ((9000, 9999), 'East_Flanders')
        ]
        for (low, high), region in ranges:
            if low <= pc <= high:
                return region
        return "Unknown"
    except:
        return "Unknown"



# ============================================================
#   HIGH-PERFORMANCE PREPROCESSOR FOR YOUR REAL DATASET
# ============================================================
class FeaturePreprocessor:

    NUMERIC_FEATURES = [
        # raw
        "number_of_rooms",
        "living_area",
        "number_of_facades",
        "garden_surface",
        "terrace_surface",

        # engineered numeric
        "total_outdoor_surface",
        "outdoor_ratio",
        "luxury_score",
        "price_per_sqm",

        # location-based numeric
        "postal_median_price",
        "postal_price_per_sqm",
        "postal_rank",

        # log transforms
        "area_log",
        "garden_log",
        "terrace_log",
        "outdoor_log",

        # interaction features
        "area_per_room",
        "garden_ratio",
    ]

    CATEGORICAL_FEATURES = [
        "type_of_property",
        "subtype_of_property",
        "state_of_building",
        "region",
        "locality_name",
        "postal_code",
    ]

    BINARY_FEATURES = [
        "equipped_kitchen",
        "furnished",
        "open_fire",
        "terrace",
        "garden",
        "swimming_pool",
    ]

    TARGET = "price"


    def __init__(self):
        self.preprocessor = None
        self.postal_medians = None
        self.postal_sqm_medians = None
        self.feature_names = None
        self.is_fitted = False


    def _add_engineered_features(self, df):
        '''Adds the engineered features to the dataframe.'''
        df = df.copy()

        # ---------------------
        # Location Features
        # ---------------------
        df["region"] = df.postal_code.apply(get_region)

        df["price_per_sqm"] = df["price"] / df["living_area"].replace(0, 1)

        df["postal_median_price"] = df["postal_code"].map(self.postal_medians)
        df["postal_price_per_sqm"] = df["postal_code"].map(self.postal_sqm_medians)

        df["postal_rank"] = df["postal_median_price"].rank(method="dense")

        # ---------------------
        # Outdoor & luxury
        # ---------------------
        df["total_outdoor_surface"] = (
            df.garden_surface.fillna(0) + df.terrace_surface.fillna(0)
        )

        total = df.living_area + df.total_outdoor_surface.replace(0, 1)
        df["outdoor_ratio"] = df.total_outdoor_surface / total.replace(0, 1)

        df["luxury_score"] = (
            df.equipped_kitchen +
            df.furnished +
            df.open_fire +
            2 * df.swimming_pool +
            (df.garden_surface > 0).astype(int) +
            (df.terrace_surface > 0).astype(int)
        )

        # ---------------------
        # Log transforms
        # ---------------------
        df["area_log"] = np.log1p(df.living_area)
        df["garden_log"] = np.log1p(df.garden_surface)
        df["terrace_log"] = np.log1p(df.terrace_surface)
        df["outdoor_log"] = np.log1p(df.total_outdoor_surface)

        # ---------------------
        # Interaction features
        # ---------------------
        df["area_per_room"] = df.living_area / df.number_of_rooms.replace(0, 1)
        df["garden_ratio"] = df.garden_surface / df.living_area.replace(0, 1)

        return df


    def _create_pipeline(self):
        '''Creates the preprocessing pipeline.'''
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        binary_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0))
        ])

        return ColumnTransformer([
            ("numeric", numeric_pipeline, self.NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, self.CATEGORICAL_FEATURES),
            ("binary", binary_pipeline, self.BINARY_FEATURES),
        ])


    def fit_transform(self, df):
        ''' Fits the preprocessor and transforms the data. '''

        self.postal_medians = df.groupby("postal_code")["price"].median()
        self.postal_sqm_medians = (df["price"] / df["living_area"]).groupby(df["postal_code"]).median()

        X = self._add_engineered_features(df)
        y = np.log1p(df["price"])  # log-transform target

        self.preprocessor = self._create_pipeline()
        X_proc = self.preprocessor.fit_transform(X)

        self.feature_names = self._extract_feature_names()
        self.is_fitted = True

        return X_proc, y


    # ============================================================
    #   TRANSFORM NEW DATA
    # ============================================================
    def transform(self, df):
        '''Transforms new data using the fitted preprocessor.'''
        X = self._add_engineered_features(df)
        return self.preprocessor.transform(X)


    # ============================================================
    #   FEATURE NAMES
    # ============================================================
    def _extract_feature_names(self):
        '''Extracts feature names after preprocessing.'''
        names = list(self.NUMERIC_FEATURES)

        enc = self.preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        names.extend(enc.get_feature_names_out(self.CATEGORICAL_FEATURES))

        names.extend(self.BINARY_FEATURES)
        return names

    def save(self, path):
        '''Saves the preprocessor to a file.'''
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "postal_medians": self.postal_medians,
                "postal_sqm_medians": self.postal_sqm_medians,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            },
            path,
        )
