import pandas as pd
import os
from src.loader import DataLoader
from src.cleaner import DataCleaner
from src.transformer import FeatureTransformer

class DataPipeline:
    """Orchestrates the entire end-to-end data science pipeline."""
    def __init__(self, file_path: str):
        self.loader = DataLoader(file_path)
        self.cleaner = DataCleaner()
        self.transformer = FeatureTransformer()

    def run_pipeline(self, output_path: str | None = None) -> pd.DataFrame:
        # 1. Load
        raw_data = self.loader.load_data()

        # 2. Clean
        cleaned_data = self.cleaner.fit_transform(raw_data)

        # 3. Transform Features
        processed_data = self.transformer.fit_transform(cleaned_data)
        
        final_df = pd.DataFrame(processed_data)

        # 4. Optional: Save to processed folder
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_df.to_csv(output_path, index=False)
            print(f"Processed data successfully saved to '{output_path}'")

        return final_df

if __name__ == "__main__":
    file_path = "data/raw/sample_data.csv"
    output_path = "data/processed/clean_features.csv"
    
    pipeline = DataPipeline(file_path=file_path)
    final_features = pipeline.run_pipeline(output_path=output_path)
    
    print("\nProcessed Features Preview:")
    print(final_features.head())