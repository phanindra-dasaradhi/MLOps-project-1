import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import (
    CONFIG_PATH,
    RAW_DIR,
    RAW_FILE_PATH,
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
)
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: dict):
        try:
            self.config = config["data_ingestion"]
            self.bucket_name = self.config["bucket_name"]
            self.file_name = self.config["bucket_file_name"]
            self.train_ratio = float(self.config["train_ratio"])

            if not (0.0 < self.train_ratio < 1.0):
                raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

            os.makedirs(RAW_DIR, exist_ok=True)

            logger.info(
                f"Initialized DataIngestion | bucket={self.bucket_name}, "
                f"file={self.file_name}, train_ratio={self.train_ratio}"
            )

        except Exception:
            logger.exception("Failed to initialize DataIngestion")
            raise CustomException("Failed to initialize DataIngestion", sys)

    def download_csv_from_gcp(self) -> None:
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Downloaded gs://{self.bucket_name}/{self.file_name} → {RAW_FILE_PATH}")

        except Exception:
            logger.exception("Failed to download file from GCP")
            raise CustomException("Failed to download file from GCP", sys)

    def split_data(self) -> None:
        try:
            logger.info("Splitting data into train and test sets")

            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data,
                test_size=1 - self.train_ratio,
                random_state=42,
            )

            os.makedirs(TRAIN_DIR, exist_ok=True)
            os.makedirs(TEST_DIR, exist_ok=True)

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Saved train → {TRAIN_FILE_PATH} | test → {TEST_FILE_PATH}")

        except Exception:
            logger.exception("Failed to split data")
            raise CustomException("Failed to split data", sys)

    def run(self) -> None:
        try:
            logger.info("Starting data ingestion pipeline")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion pipeline completed successfully")

        except CustomException as ce:
            logger.error(str(ce))
            raise  # re-raise so failures are visible to callers/CI

        finally:
            logger.info("Data ingestion pipeline execution finished")


if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    DataIngestion(config).run()
