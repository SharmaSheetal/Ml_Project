import os
import sys
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using the dill library.

    Args:
        file_path (str): Path to save the object.
        obj (Any): The Python object to save.

    Raises:
        CustomException: If an error occurs during the saving process.
    """
    try:
        # Create the directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file path
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple machine learning models using GridSearchCV and calculate r2 scores.

    Args:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test target values.
        models (dict): Dictionary of models to evaluate.
        params (dict): Dictionary of hyperparameters for each model.

    Returns:
        dict: A dictionary with model names as keys and their test r2 scores as values.

    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        logging.info("Evaluating models")
        report = {}

        # Iterate over the models and their respective parameters
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[model_name]

            # Perform GridSearchCV to find the best hyperparameters
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            # Update the model with the best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on training and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate r2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Add the test score to the report dictionary
            report[model_name] = test_model_score

        logging.info("Models evaluated successfully")
        return report

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using the pickle library.

    Args:
        file_path (str): Path to the file containing the object.

    Returns:
        Any: The loaded Python object.

    Raises:
        CustomException: If an error occurs during the loading process.
    """
    try:
        # Open and load the object from the specified file path
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)
