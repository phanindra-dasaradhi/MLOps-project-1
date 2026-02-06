# import os
# import pandas as pd
# import numpy as np
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from utils.common_functions import read_yaml, load_data
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# import yaml

# logger = get_logger(__name__)

# class DataPreprocessor:

#     def __init__(self, train_path, test_path, processed_dir, config_path):
#         self.train_path = train_path
#         self.test_path = test_path
#         self.processed_dir = processed_dir  
#         self.config = read_yaml(config_path)

#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir, exist_ok=True)
#             logger.info(f"Created processed data directory at {self.processed_dir}")

#     def preprocess_data(self, data):
#         try:
#             logger.info("Starting data preprocessing")
#             logger.info("Dropping collumns")
#             data.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True, errors='ignore')
#             data.drop_duplicates(inplace=True)

#             cat_cols = self.config['data_processing']['categorical_columns']
#             num_cols = self.config['data_processing']['numerical_columns']  

#             logger.info("Applying Label Encoding")
#             label_encoders = {}
#             mappings = {}
#             for col in cat_cols:
#                 data[col] = label_encoders.fit_transform(data[col])
#                 mappings[col] = {label:code for label, code in zip(label_encoders.classes_, label_encoders.transform(label_encoders.classes_))}

#             logger.info("Label Mappings:")
#             for col, mapping in mappings.items():
#                 logger.info(f"{col}: {mapping}")
#             logger.info("Doing skewness handling")

#             skew_threshold = self.config['data_processing']['skewness_threshold']
#             skewness = data[num_cols].apply(lambda x: x.skew())

#             for col in skewness[skewness > skew_threshold].index:
#                 data[col] = np.log1p(data[col])
#                 logger.info(f"Applied log transformation to {col} due to skewness of {skewness[col]:.2f}")
#                 return data
#         except Exception as e:
#             logger.error("Error during data preprocessing")
#             raise CustomException("Failed to preprocess data", e)
        
#         def balance_data(self, data):
#             try:
#                 logger.info("Starting data balancing using SMOTE")
#                 X = data.drop(columns=['booking_status'])
#                 y = data['booking_status']

#                 smote = SMOTE(random_state=42)
#                 X_resampled, y_resampled = smote.fit_resample(X, y)

#                 balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
#                 balanced_data['Target'] = y_resampled

#                 logger.info(f"Data balanced using SMOTE. Original samples: {len(data)}, Resampled samples: {len(balanced_data)}")
#                 return balanced_data
#             except Exception as e:
#                 logger.error("Error during data balancing")
#                 raise CustomException("Failed to balance data", e)
        
#         def select_features(self, data):
#             try:
#                 logger.info("Starting feature selection using Random Forest")
#                 X = data.drop(columns=['booking_status'])
#                 y = data['booking_status']

#                 model = RandomForestClassifier(random_state=42)
#                 model.fit(X, y)

#                 Feature_importance = model.feature_importances_
#                 feature_importance_df = pd.DataFrame({
#                     'Feature': X.columns,
#                     'Importance': Feature_importance
#                 }).sort_values(by='Importance', ascending=False)
                
#                 top_features_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#                 num_features_to_select = self.config['data_processing']['no_of_features']

#                 top_10_features = top_features_importance_df["Feature"].head(num_features_to_select).values
#                 logger.info(f"Top {num_features_to_select} features selected: {top_10_features}")
#                 top_10_df = data[top_10_features.tolist() + ["booking_status"]]

#                 logger.info(f"Selected top {num_features_to_select} features based on importance")
#                 return top_10_df
#             except Exception as e:
#                 logger.error("Error during feature selection")
#                 raise CustomException("Failed to select features", e)
            

#         def save_processed_data(self, data, file_path):
#             try:
#                 logger.info(f"Saving processed data to {file_path}")
#                 data.to_csv(file_path, index=False)
#                 logger.info(f"Processed data saved to {file_path}")
#             except Exception as e:
#                 logger.error("Error saving processed data")
#                 raise CustomException("Failed to save processed data", e)
                
#         def process(self):
#             try:
#                 logger.info("Loading data from RAW directory")
#                 train_data = load_data(self.train_path)
#                 test_data = load_data(self.test_path)

#                 train_data = self.preprocess_data(train_data)
#                 test_data = self.preprocess_data(test_data)

#                 train_data = self.balance_data(train_data)
#                 test_data = self.balance_data(test_data)

#                 train_data = self.select_features(train_data)
#                 test_data = test_data[train_data.columns]

#                 self.save_data(train_data, PROCESSED_TRAIN_DATA_PATH)
#                 self.save_processed_data(test_data, PROCESSED_TEST_DATA_PATH)

#                 logger.info("Data preprocessing completed successfully")
#             except Exception as e:
#                 logger.error("Error during data preprocessing pipeline")
#                 raise CustomException("Failed to run data preprocessing pipeline", e)
            
# if __name__ == "__main__":
#         processor = DataPreprocessor(
#             train_path=TRAIN_FILE_PATH,
#             test_path=TEST_FILE_PATH,
#             processed_dir=PROCESSED_DIR,
#             config_path=CONFIG_PATH
#         )
#         processor.process()

import os
import sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        try:
            self.train_path = train_path
            self.test_path = test_path
            self.processed_dir = processed_dir  
            self.config = read_yaml(config_path)

            if not os.path.exists(self.processed_dir):
                os.makedirs(self.processed_dir, exist_ok=True)
                logger.info(f"Created processed data directory at {self.processed_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, data):
        try:
            logger.info("Starting data preprocessing")
            data = data.copy()
            data.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True, errors='ignore')
            data.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']  

            le = LabelEncoder()
            for col in cat_cols:
                if col in data.columns:
                    data[col] = le.fit_transform(data[col])

            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = data[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > skew_threshold].index:
                data[col] = np.log1p(data[col])
            
            return data
        except Exception as e:
            raise CustomException(e, sys)
        
    def balance_data(self, data):
        try:
            logger.info("Balancing data...")
            target_col = 'booking_status'
            X = data.drop(columns=[target_col])
            y = data[target_col]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_data[target_col] = y_resampled
            return balanced_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def select_features(self, data):
        try:
            logger.info("Selecting features...")
            target_col = 'booking_status'
            X = data.drop(columns=[target_col])
            y = data[target_col]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            importances = importances.sort_values(by='Importance', ascending=False)
            
            num_features = self.config['data_processing']['no_of_features']
            top_features = importances["Feature"].head(num_features).tolist()
            
            return data[top_features + [target_col]]
        except Exception as e:
            raise CustomException(e, sys)

    def save_processed_data(self, data, file_path):
        try:
            data.to_csv(file_path, index=False)
        except Exception as e:
            raise CustomException(e, sys)
                
    def process(self):
        try:
            logger.info("Loading data...")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            train_df = self.select_features(train_df)
            
            # Match columns
            test_df = test_df[train_df.columns]

            self.save_processed_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_processed_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Preprocessing successful!")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        processor = DataPreprocessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH
        )
        processor.process()
    except Exception as e:
        print(e)