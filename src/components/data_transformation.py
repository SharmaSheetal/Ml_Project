import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation, specifying paths for artifacts.
    """
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformer:
    """
    This class handles all operations related to data transformation, including
    creating preprocessing pipelines and transforming train/test datasets.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline for numerical and categorical features.

        Steps for Numerical Pipeline:
        - Handle missing values using median imputation.
        - Scale the data using StandardScaler.

        Steps for Categorical Pipeline:
        - Handle missing values using most frequent imputation.
        - One-hot encode the categorical features.
        - Scale the one-hot encoded data using StandardScaler (with_mean=False).

        Returns:
            ColumnTransformer: A preprocessing pipeline for numerical and categorical features.
        """
        try:
            numerical_column = ["writing_score", "reading_score"]
            categorical_column = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            logging.info(f"Categorical Columns: {categorical_column}")
            logging.info(f"Numerical Columns: {numerical_column}")

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_column),
                    ("cat_pipeline", cat_pipeline, categorical_column)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test datasets, applies the preprocessing pipeline, and saves the preprocessor object.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            tuple: Transformed train and test arrays, and the preprocessor object file path.
        """
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data reading completed")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target variable for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Performing data transformation on train and test dataframes")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate transformed features with target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor object")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
