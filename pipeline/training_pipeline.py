from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *




if __name__ == "__main__":
    #### DATA Ingestion ####
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    #### DATA Processing ####
    processor = DataPreprocessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH
        )
    processor.process()

    #### MODEL Training ####
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()

