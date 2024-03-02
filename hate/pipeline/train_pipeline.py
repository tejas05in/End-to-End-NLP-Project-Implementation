import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_ingestion import DataIngestion
from hate.components.data_validation import DataValidation
from hate.components.data_transformation import DataTransformation
from hate.components.model_trainer import ModelTrainer
from hate.entity.config_entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataTransformationConfig,
                                       ModelTrainerConfig)
from hate.entity.artifact_entity import (DataIngestionArtifacts,
                                         DataValidationArtifacts,
                                         DataTransformationArtifacts,
                                         ModelTrainerArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

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

    def start_data_transformation(self, data_ingestion_artifacts=DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info(
            "Entering data transformation method of the TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifacts=data_ingestion_artifacts)
            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info(
                "Exited the start_data_transformation method of the TrainPipeline class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info(
            "Entering model trainer method of the TrainPipeline class")
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifacts=data_transformation_artifacts)
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info(
                "Exited the start_model_trainer method of the TrainPipeline class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        logging.info(
            "Entered the run_pipeline method of the TrainPipeline class.")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(imbalanced_data_path=data_ingestion_artifacts.imbalanced_data_file_path,
                                                                   raw_data_path=data_ingestion_artifacts.raw_data_file_path)
            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifacts
            )
            mode_trainer_artifacts = self.start_model_trainer(
                data_transformation_artifacts=data_transformation_artifacts
            )
            logging.info(
                "Exited the run_pipeline method of the TrainPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e
