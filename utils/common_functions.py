# import os
# import pandas as pd
# from src .logger import get_logger
# from src.custom_exception import CustomException
# import yaml

# logger = get_logger(__name__)

# def read_yaml(file_path):
#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File is not in the given path")

#         with open(file_path, "r") as yaml_file:
#             config = yaml.safe_load(yaml_file)
#             logger.info("YAML file loaded successfully")
#             return config

#     except Exception as e:
#         logger.error(f"Error reading YAML file")
#         raise CustomException("Failed to read YAML file", e)

# def load_data(path):
#     try:
#         logger.info(f"Loading data from {path}")
#         data = pandas.read_csv(path)
#         logger.info(f"Data loaded successfully from {path}")
#         return data
#     except Exception as e:
#         logger.error(f"Error loading data from {path}")
#         raise CustomException("Failed to load data", e)
            
import os
import sys
import pandas as pd
import yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at: {os.path.abspath(file_path)}")

        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from {file_path}")
            return config

    except Exception as e:
        logger.error(f"Error reading YAML file: {file_path}")
        # Always pass 'sys' as the second argument
        raise CustomException(e, sys)

def load_data(path):
    try:
        logger.info(f"Attempting to load data from: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found at: {os.path.abspath(path)}")
            
        # Fixed: Changed 'pandas' to 'pd' to match your import
        data = pd.read_csv(path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {path}")
        # Always pass 'sys' as the second argument
        raise CustomException(e, sys)