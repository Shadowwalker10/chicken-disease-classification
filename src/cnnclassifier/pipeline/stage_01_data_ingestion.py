from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.data_ingestion import DataIngestion
from cnnclassifier.logging import logger
from cnnclassifier.utils.common import get_size
import os
import shutil
from pathlib import Path
import sys
import json


## Creating the pipeline
class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        dataset_ingestion = DataIngestion(config = data_ingestion_config)
        main_path = data_ingestion_config.local_data_file
        to_check = Path(os.path.join(main_path, "Train"))
        if "Train"  in os.listdir(main_path) and get_size(to_check)>100:
            logger.info(f"File : {data_ingestion_config.dataset_name} already downloaded")
            return
        logger.info(f"Downloading the dataset {data_ingestion_config.dataset_name}")
        dataset_ingestion.download_file()
        logger.info(f"Dataset: {data_ingestion_config.dataset_name} successfully downloaded")

        source_dir = Path(os.path.join(data_ingestion_config.local_data_file, 
                                       os.listdir(data_ingestion_config.local_data_file)[0]))
        
        image_files = os.listdir(source_dir)
        logger.info("Rearranging the dataset folder structure")
        for img in image_files:
            class_name = img.split(".")[0]
            ##create class directory if not exists
            class_directory = source_dir/class_name

            if not class_directory.exists():
                class_directory.mkdir()
            ## Move the image to class directory
            
            source_path = source_dir/img
            destination_path = class_directory/img
            shutil.move(src=str(source_path), dst=str(destination_path))
        
        logger.info("Successfully restructured dataset folder structure")








