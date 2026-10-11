import pytest
import pandas as pd
import numpy as np
from src.loader import DataLoader
from src.cleaner import DataCleaner
from src.transformer import FeatureTransformer

@pytest.fixture
def sample_raw_df():
    """Fixture providing a raw dataframe with missing values and duplicates."""
    return pd.DataFrame({
        'age': [25, 32, None, 32, 25],
        'income': [50000, 80000, 120000, 80000, 50000],
        'city': ['New York', 'London', 'Paris', 'London', 'New York'],
        'purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
    })

def test_data_cleaner(sample_raw_df):
    """Test that DataCleaner removes duplicates and imputes missing values."""
    cleaner = DataCleaner()
    cleaned_df = cleaner.fit_transform(sample_raw_df)
    
    # Ensure it's a DataFrame
    assert isinstance(cleaned_df, pd.DataFrame), "Output must be a pandas DataFrame."

    # 1. Check duplicates are dropped (5 rows -> 3 unique rows)
    assert len(cleaned_df) == 3, f"Expected 3 rows after duplicate removal, got {len(cleaned_df)}"
    
    # 2. Check no missing values remain
    assert cleaned_df.isnull().sum().sum() == 0, "Found remaining missing values after cleaning."

def test_feature_transformer(sample_raw_df):
    """Test that FeatureTransformer scales numerics and one-hot encodes categoricals."""
    # Clean first so transformer receives clean data
    cleaner = DataCleaner()
    cleaned_df = cleaner.fit_transform(sample_raw_df)
    
    transformer = FeatureTransformer()
    # Explicitly cast to pandas DataFrame for static typing and test assertions
    transformed_output = transformer.fit_transform(cleaned_df)
    transformed_df = pd.DataFrame(transformed_output)
    
    # 1. Check that original categorical columns are dropped and replaced with encoded ones
    assert 'city' not in transformed_df.columns, "Original categorical column 'city' should be dropped."
    assert 'city_London' in transformed_df.columns, "Expected encoded column 'city_London' to be present."
    
    # 2. Check that numeric columns are scaled (mean approx 0)
    assert np.isclose(transformed_df['age'].mean(), 0.0, atol=1e-7), "Numeric column 'age' was not properly scaled."
    
    # 3. Check output is a DataFrame and has no nulls
    assert isinstance(transformed_df, pd.DataFrame), "Output must be a pandas DataFrame."
    assert transformed_df.isnull().sum().sum() == 0, "Transformed dataframe contains nulls."

def test_loader_exception(tmp_path):
    """Test that DataLoader raises an exception when file does not exist."""
    fake_path = tmp_path / "non_existent.csv"
    loader = DataLoader(str(fake_path))
    
    with pytest.raises(Exception):
        loader.load_data()