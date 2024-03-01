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