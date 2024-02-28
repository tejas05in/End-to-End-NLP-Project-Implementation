from dataclasses import dataclass

# Data Ingestion Artifacts

@dataclass
class DataIngestionArtifacts:
    """Data Ingestion Artifacts"""
    imbalanced_data_file_path: str
    raw_data_file_path: str