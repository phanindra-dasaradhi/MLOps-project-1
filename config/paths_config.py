import os

################# DATA INGESTION ################

ARTIFACT_DIR = "artifacts"

RAW_DIR = os.path.join(ARTIFACT_DIR, "raw")
TRAIN_DIR = os.path.join(ARTIFACT_DIR, "train")
TEST_DIR = os.path.join(ARTIFACT_DIR, "test")

RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(TRAIN_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(TEST_DIR, "test.csv")

################# CONFIG ################

CONFIG_PATH = "config/config.yaml"



################# DATA PREPROCESSING ################

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train_processed.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "test_processed.csv")


################# MODEL TRAINING ################
MODEL_OUTPUT_PATH = "artifacts/model/lgbm_model.pkl"