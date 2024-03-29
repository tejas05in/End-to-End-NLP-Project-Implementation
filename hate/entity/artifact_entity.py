from dataclasses import dataclass

# Data Ingestion Artifacts


@dataclass
class DataIngestionArtifacts:
    """Data Ingestion Artifacts"""
    imbalanced_data_file_path: str
    raw_data_file_path: str


@dataclass
class DataValidationArtifacts:
    """Data Validation Artifacts"""
    result_file_path: str


@dataclass
class DataTransformationArtifacts:
    """Data Transformation Artifacts"""
    transformed_data_path: str


@dataclass
class ModelTrainerArtifacts:
    """Model Trainer Artifacts"""
    trained_model_path: str
    x_test_path: list
    y_test_path: list


@dataclass
class ModelEvaluationArtifacts:
    """Model Evaluation Artifacts"""
    is_model_accepted: bool


@dataclass
class ModelPusherArtifacts:
    bucket_name: str
