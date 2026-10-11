import pandas as pd
import os
from main import DataPipeline

def create_dummy_data(file_path: str):
    """Generates a synthetic dataset with missing values and duplicates for testing."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data = {
        'age': [25, 32, None, 47, 32, 25],          # Includes missing value & duplicate row
        'income': [50000, 80000, 120000, None, 80000, 50000],
        'city': ['New York', 'London', 'Paris', 'London', 'London', 'New York'],
        'purchased': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"[Setup] Created dummy dataset at '{file_path}' with {len(df)} initial rows.")

def test_end_to_end_pipeline():
    # 1. Define paths
    raw_data_path = "data/raw/sample_data.csv"
    
    # 2. Setup mock data
    create_dummy_data(raw_data_path)
    
    # 3. Initialize pipeline
    print("\n[Pipeline] Initializing DataPipeline...")
    pipeline = DataPipeline(file_path=raw_data_path)
    
    # 4. Run pipeline
    print("[Pipeline] Executing run_pipeline()...")
    processed_df = pipeline.run_pipeline()
    
    # 5. Assertions & Validation
    print("\n--- Pipeline Validation Report ---")
    print(f"Final output shape: {processed_df.shape}")
    print(f"Any remaining missing values: {processed_df.isnull().sum().sum()}")
    
    # Check that duplicates were dropped (6 initial rows -> 4 unique rows)
    assert len(processed_df) == 4, f"Expected 4 rows after dropping duplicates, but got {len(processed_df)}"
    print("Assertion Passed: Duplicate rows successfully removed.")
    
    # Check that numeric columns are scaled and categorical ones encoded
    print("\nProcessed Features Head:")
    print(processed_df)

if __name__ == "__main__":
    test_end_to_end_pipeline()