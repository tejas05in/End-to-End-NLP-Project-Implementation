from dataclasses import dataclass
from hate.constants import *
import os


@dataclass
class DataIngestionConfig:
    """
    DataIngestionConfig is a dataclass that holds the configuration for data ingestion.
    """

    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
            os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_ARTIFACTS_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_IMBALANCE_DATA_DIR)
        self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_RAW_DATA_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(
            self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)


@dataclass
class DataValidationConfig:
    """
    DataValidationConfig is a dataclass that holds the configuration for data validation.
    """

    def __init__(self):
        self.DATA_VALIDATION_ARTIFACTS_DIR: str = os.path.join(
            os.getcwd(), ARTIFACTS_DIR, DATA_VALIDATION_ARTIFACTS_DIR)
        self.RESULT_FILE_PATH = os.path.join(
            self.DATA_VALIDATION_ARTIFACTS_DIR, DATA_VALIDATION_FILE_NAME)

@dataclass
class DataTransformationConfig:
    """
    DataTransformationConfig is a dataclass that holds the configuration for data transformation.
    """
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(
            os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET
        

@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.TRINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR , MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRINED_MODEL_PATH = os.path.join(self.TRINED_MODEL_DIR, TRAINER_MODEL_NAME)
        self.X_TEST_DATA_PATH = os.path.join(self.TRINED_MODEL_DIR, X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH = os.path.join(self.TRINED_MODEL_DIR, Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH = os.path.join(self.TRINED_MODEL_DIR, X_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.RANDOM_STATE = RANDOM_STATE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT