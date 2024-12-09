import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for storing the file path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    A class to train and evaluate machine learning models, 
    select the best-performing model, and save it for later use.
    """
    def __init__(self):
        """
        Initializes the ModelTrainer class with a configuration object.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Initiates the model training process by splitting the data, 
        training multiple models, evaluating their performance, 
        and saving the best model based on R² score.

        Args:
            train_array (numpy.ndarray): Training data array.
            test_array (numpy.ndarray): Testing data array.
            preprocessor_path (str): Path to the preprocessor object.

        Returns:
            float: R² score of the best-performing model on the test set.

        Raises:
            CustomException: If no model achieves a satisfactory score.
        """
        try:
            # Split data into features and target for training and testing
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],   # Target for training
                test_array[:, :-1],   # Features for testing
                test_array[:, -1]     # Target for testing
            )

            # Define models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate all models and generate a report with their performance
            logging.info("Evaluating models with hyperparameters")
            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params
            )

            # Identify the best-performing model based on the R² score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Check if the best model meets the minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with acceptable performance")

            logging.info(f"Best model ({best_model_name}) selected with R² score: {best_model_score}")

            # Save the best model to the specified file path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model and calculate the R² score on test data
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # Handle exceptions and log the error
            raise CustomException(e, sys)
