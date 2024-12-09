import pandas as pd
import numpy as np
import sys
import os
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

class PredictionPipeline:
    """
    Pipeline for loading the trained model and preprocessor, 
    transforming input features, and generating predictions.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the target value(s) based on the input features.

        Args:
            features (pd.DataFrame): A dataframe containing input features.

        Returns:
            np.ndarray: Array of predicted values.

        Raises:
            CustomException: If an error occurs during prediction.
        """
        try:
            # Paths for the saved model and preprocessor objects
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info("Loading model and preprocessor")
            
            # Load the trained model and preprocessor from the specified paths
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            logging.info("Model and preprocessor loaded successfully")

            # Transform input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Generate predictions using the trained model
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            # Log and raise a custom exception in case of errors
            raise CustomException(e, sys)


class CustomData:
    """
    Class to encapsulate custom input data and convert it into a format 
    suitable for prediction.
    """
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        """
        Initializes the CustomData class with individual input features.

        Args:
            gender (str): Gender of the student.
            race_ethnicity (str): Race/ethnicity of the student.
            parental_level_of_education (str): Parent's level of education.
            lunch (str): Type of lunch the student has.
            test_preparation_course (str): Test preparation course status.
            reading_score (int): Reading score of the student.
            writing_score (int): Writing score of the student.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the input features into a pandas DataFrame for prediction.

        Returns:
            pd.DataFrame: DataFrame containing the input features.

        Raises:
            CustomException: If an error occurs during DataFrame creation.
        """
        try:
            # Create a dictionary of input features
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary into a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Log and raise a custom exception in case of errors
            raise CustomException(e, sys)
