import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Responsible for loading raw data from a source (CSV, Database, API)."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading data from {self.file_path}...")
            df = pd.read_csv(self.file_path)
            logging.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise e