import sys
import os
import pandas as pd
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for defining paths used during data ingestion.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    """
    Class to handle data ingestion, including reading the dataset and splitting it into training and testing sets.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads raw data, saves it to artifacts, and splits it into training and test sets.

        Returns:
            tuple: Paths to the training and test datasets.
        """
        logging.info("Entered the data ingestion method")
        try:
            # Reading the dataset into a DataFrame
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("Reading the dataset as a DataFrame")

            # Creating the directory for storing artifacts if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Splitting the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the training and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initializing and executing the data ingestion process
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    # Initializing and executing the data transformation process
    data_transformation = DataTransformer()
    data_transformation.initiate_data_transformation(train_path, test_path)
