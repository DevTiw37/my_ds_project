import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Responsible for scaling numerical features and encoding categorical variables."""
    def __init__(self):
        self.scaler = StandardScaler()
        # Enforce dense output so it never returns a sparse matrix
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        logging.info("Fitting FeatureTransformer...")
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Transforming features...")
        df = X.copy()
        transformed_dfs: list[pd.DataFrame] = [df]

        # Scale numerical features
        if self.numeric_cols:
            scaled_array = self.scaler.transform(df[self.numeric_cols])
            scaled_df = pd.DataFrame(scaled_array, columns=self.numeric_cols, index=df.index)
            transformed_dfs.append(scaled_df)

        # Encode categorical features
        if self.categorical_cols:
            encoded_output = self.encoder.transform(df[self.categorical_cols])
            # Safely cast to a dense numpy array without risking linter attribute complaints
            encoded_array = np.asarray(encoded_output)
            encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
            transformed_dfs.append(encoded_df)

        # Drop original columns and concatenate transformed ones
        drop_cols = self.numeric_cols + self.categorical_cols
        base_df = df.drop(columns=drop_cols)
        
        result_df = pd.concat([base_df] + [d for d in transformed_dfs if d is not df], axis=1)

        logging.info(f"Feature transformation complete. Output shape: {result_df.shape}")
        return result_df