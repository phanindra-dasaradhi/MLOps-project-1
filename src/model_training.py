import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn



logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path} and {self.test_path}")
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)
            
            X_train = train_data.drop(columns=['booking_status'])
            y_train = train_data['booking_status']
            X_test = test_data.drop(columns=['booking_status'])
            y_test = test_data['booking_status']

            logger.info("Data loaded and split into features and target variable")
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(f"Error in loading and splitting data: {e}")
        
    def train_model(self, X_train, y_train):
        try:
            logger.info("Starting model training with RandomizedSearchCV")
            lgbm_model = LGBMClassifier(random_state=self.random_search_params.get('random_state', 42))

            logger.info("Performing hyperparameter tuning")
                
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params.get('n_iter'),
                cv=self.random_search_params.get('cv'),
                scoring=self.random_search_params.get('scoring'),
                random_state=self.random_search_params.get('random_state'),
                n_jobs=self.random_search_params.get('n_jobs'),
                verbose=self.random_search_params.get('verbose')
            )

            logger.info("Starting the model training process")  
                
            random_search.fit(X_train, y_train)
                
            logger.info("Hyperparameter tuning completed")
            logger.info(f"Best parameters found: {random_search.best_params_}")
                
            best_lgbm_model = random_search.best_estimator_
            return best_lgbm_model
        except Exception as e:
            raise CustomException(f"Error in model training: {e}")

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the trained model")
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            
            logger.info(f"Model Evaluation Metrics: Accuracy: {accuracy},\n Precision: {precision},\n Recall: {recall},\n F1-Score: {f1}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise CustomException(f"Error in model evaluation: {e}")
    
    def save_model(self, model):
        try:
            logger.info(f"Saving the trained model to {self.model_output_path}")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Model directory created successfully")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error in saving the model: {e}")
            raise CustomException(f"Error in saving the model: {e}")
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training pipeline")
                logger.info("Starting MLflow experiment tracking")
                logger.info("logging the training and testing data paths as parameters")
                mlflow.log_artifact(self.train_path, artifact_path="dataset")
                mlflow.log_artifact(self.test_path, artifact_path="dataset")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_model(X_train, y_train)
                evaluation_metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                mlflow.log_metrics(evaluation_metrics)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                mlflow.log_metrics(metrics)
                self.save_model(best_lgbm_model)

                logger.info("Logging model in MLflow")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")

                logger.info("Logging model parameters and metrics in MLflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training pipeline completed successfully")

                return evaluation_metrics
        except Exception as e:
            logger.error(f"Error in running the model training pipeline: {e}")
            raise CustomException(f"Error in running the model training pipeline: {e}")
        
if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()
    logger.info("Model training process completed successfully")