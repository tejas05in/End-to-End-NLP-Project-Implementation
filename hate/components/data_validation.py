import os
import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataValidationConfig
from hate.entity.artifact_entity import DataValidationArtifacts
import pandas as pd


class DataValidation:
    """
    Class for data validation.
    """

    def __init__(self, data_validation_config: DataValidationConfig, imbalanced_data_path, raw_data_path):
        self.data_validation_config = data_validation_config
        self.imbalanced_data_path = imbalanced_data_path
        self.raw_data_path = raw_data_path

    def validate_data(self):
        try:
            logging.info(
                "Entered into the validate_data method of the DataValidation class")
            os.makedirs(
                self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR, exist_ok=True)
            imbalanced_data = pd.read_csv(self.imbalanced_data_path)
            raw_data = pd.read_csv(self.raw_data_path)
            imbalanced_columns = ['id', 'label', 'tweet']
            raw_columns = ['Unnamed: 0', 'count', 'hate_speech',
                           'offensive_language', 'neither', 'class', 'tweet']
            imb_result = None
            raw_result = None
            if list(imbalanced_data.columns) == imbalanced_columns:
                imb_result = True
            else:
                imb_result = False
            if list(raw_data.columns) == raw_columns:
                raw_result = True
            else:
                raw_result = False
            logging.info(
                f"Imbalanced data validation result: {imb_result} & raw data validation result: {raw_result}")
            logging.info(
                "Exited the validate_data method of the DataValidation class")
            return imb_result, raw_result
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e

    def initiate_data_validation(self):
        logging.info(
            "Entering the initiate data validation method of the DataValidation class")
        try:
            imb_result, raw_result = self.validate_data()
            with open(self.data_validation_config.RESULT_FILE_PATH, 'w') as result_file:
                result_file.write(
                    f"The imbalanced data validation result is: {imb_result} \n")
                result_file.write(
                    f"The raw data validation result is: {raw_result}")
                data_validation_artifacts = DataValidationArtifacts(
                    result_file_path=self.data_validation_config.RESULT_FILE_PATH
                )
                logging.info(
                    f"Data Validation artifact: {data_validation_artifacts}")
                logging.info(
                    "Exited the initiate data validation method of the DataValidation class")
                return data_validation_artifacts
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys) from e
