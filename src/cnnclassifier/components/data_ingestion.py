## Update Components
import os
import zipfile
from cnnclassifier.logging import logger
from kaggle.api.kaggle_api_extended import KaggleApi
from cnnclassifier.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.api = KaggleApi()
        self.api.authenticate()

    def download_file(self):
        logger.info(f"Downloading Dataset: {self.config.dataset_name} to {self.config.local_data_file}")
        # Extract the dataset
        dataset_name = self.config.dataset_name
        output_path = self.config.local_data_file

        ## Download the dataset from kaggle
        self.api.dataset_download_files(dataset = dataset_name, path = output_path, unzip = True)
        logger.info(f"{dataset_name} extracted successfully to {output_path}")

