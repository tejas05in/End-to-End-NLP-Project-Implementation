import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_ingestion import DataIngestion
from hate.components.data_validation import DataValidation
from hate.entity.config_entity import (DataIngestionConfig,
                                       DataValidationConfig)
from hate.entity.artifact_entity import (DataIngestionArtifacts,
                                         DataValidationArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info(
            "Entered the start_data_ingestion method of the TrainPipeline class.")
        try:
            logging.info("Getting the data from GCLoud Storage bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from GCLoud Storage")
            logging.info(
                "Exited the start_data_ingestion method of the TrainPipeline class")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_validation(self, imbalanced_data_path, raw_data_path) -> DataValidationArtifacts:
        logging.info(
            "Entered start_data_validation method of the TrainPipeline class")
        try:
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                imbalanced_data_path=imbalanced_data_path,
                raw_data_path=raw_data_path)
            data_validation_artifacts = data_validation.initiate_data_validation()
            logging.info(
                "Exited the start_data_validation method of the TrainPipeline class")
            return data_validation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        logging.info(
            "Entered the run_pipeline method of the TrainPipeline class.")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(imbalanced_data_path=data_ingestion_artifacts.imbalanced_data_file_path,
                                                                   raw_data_path=data_ingestion_artifacts.raw_data_file_path)
            logging.info(
                "Exited the run_pipeline method of the TrainPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e
