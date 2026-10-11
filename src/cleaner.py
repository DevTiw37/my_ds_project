import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

class DataCleaner(BaseEstimator, TransformerMixin):
    """Responsible for handling missing values, duplicates, and data type corrections."""
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless cleaner, nothing to fit
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning...")
        df = X.copy()

        # 1. Drop duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        logging.info(f"Dropped {initial_rows - len(df)} duplicate rows.")

        # 2. Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'str']).columns

        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        if len(categorical_cols) > 0:
            df[categorical_cols] = df[categorical_cols].fillna("Missing")

        logging.info("Data cleaning completed.")
        return df