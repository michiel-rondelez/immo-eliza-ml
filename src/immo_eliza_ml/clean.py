import pandas as pd
import numpy as np


class CleanData:
    def __init__(self):
        pass

    def check_missing(self, df):
        missing_count = df.isna().sum()
        missing_percent = (df.isna().mean() * 100).round(2)
        missing_table = pd.DataFrame({
            "missing_count": missing_count,
            "missing_percent": missing_percent
        }).sort_values("missing_percent", ascending=False)
        print(missing_table)

    def clean(self, df):
        df = df.copy()
        df = self._rename_columns(df)
        df = self._drop_property_id_column(df)
        df = self._drop_rows_with_missing_price(df)
        df = self._drop_rows_where_all_elements_missing(df)
        df = self._remove_duplicates(df)
        df = self._drop_columns_derived_from_other_columns(df)
        df = self._lowercase_columns_with_strings(df)
        df = self._convert_booleans(df)
        df = self._impute_missing(df)
        df = self._clip_outliers(df)
        df = self._clean_object_columns(df)
        df.head()
        return df

    def _rename_columns(self, df):
        return df.rename(columns=lambda x: x.lower().strip().replace(" ", "_"))

    def _drop_property_id_column(self, df):
        if "property_id" in df.columns:
            df = df.drop(columns=["property_id"])
        return df

    def _lowercase_columns_with_strings(self, df):
        object_cols = df.select_dtypes(include="object").columns
        for col in object_cols:
            df[col] = df[col].str.lower().str.strip()
        return df

    def _drop_rows_with_missing_price(self, df):
        if "price" in df.columns:
            df = df.dropna(subset=["price"])
        return df

    def _drop_columns_derived_from_other_columns(self, df):
        if "price_per_m2" in df.columns:
            df = df.drop(columns=["price_per_m2"])
        return df

    def _drop_rows_where_all_elements_missing(self, df):
        df = df.dropna(how="all")
        return df

    def _remove_duplicates(self, df):
        df = df.drop_duplicates()
        return df

    def _convert_booleans(self, df):
        bool_cols = ["garden", "terrace", "furnished", "open_fire", "swimming_pool"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def _impute_missing(self, df):
        df.loc[(df["garden"] == 0) & (df["garden_surface"].isna()), "garden_surface"] = 0
        df.loc[(df["garden"] == 1) & (df["garden_surface"].isna()),
               "garden_surface"] = df.loc[df["garden"] == 1, "garden_surface"].median()
        df.loc[(df["terrace"] == 0) & (df["terrace_surface"].isna()), "terrace_surface"] = 0
        df.loc[(df["terrace"] == 1) & (df["terrace_surface"].isna()),
               "terrace_surface"] = df.loc[df["terrace"] == 1, "terrace_surface"].median()
        df["number_of_facades"] = df.groupby("type_of_property")["number_of_facades"]\
                                    .transform(lambda x: x.fillna(x.median()))
        df["living_area"] = df.groupby("subtype_of_property")["living_area"]\
                              .transform(lambda x: x.fillna(x.median()))
        df["living_area"] = df["living_area"].fillna(df["living_area"].median())
        df["number_of_rooms"] = df["number_of_rooms"].fillna(df["number_of_rooms"].median())
        return df

    def _clip_outliers(self, df):
        cols_to_clip = ["price", "living_area", "garden_surface", "terrace_surface"]
        for col in cols_to_clip:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q99)
        return df
    
    def _clean_object_columns(self, df):
        obj_cols = df.select_dtypes(include="object").columns
        
        for col in obj_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r"\s+", "_", regex=True)   # replace any whitespace
                .replace({"": "unknown", "nan": "unknown"})
                .fillna("unknown")
            )
        return df
    