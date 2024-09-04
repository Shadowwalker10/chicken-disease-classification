from cnnclassifier.logging import logger
from pathlib import Path
import os
import json
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnclassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnclassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnclassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"<<<<<{STAGE_NAME} Started>>>>>")
    obj = DataIngestionPipeline()
    obj.main()
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    source_dir = Path(os.path.join(data_ingestion_config.local_data_file, 
                                os.listdir(data_ingestion_config.local_data_file)[0]))
    class_image_counts = {}

    cnt = 0
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            num_images = len(list(class_dir.glob("*.jpg")))
            class_image_counts[cnt] = num_images
        cnt+=1

    logger.info(f"Image Count per class: {class_image_counts}")

    ## Calculate total number of images
    total_images = sum(class_image_counts.values())

    class_weights = {class_name: total_images/count for class_name, count in class_image_counts.items()}
    
    logger.info(f"Class Weight for each class: {class_weights}")

    with open(data_ingestion_config.class_weight, "w") as file:
        json.dump(class_weights, file)

    logger.info(f"Class weights successfully saved to: {data_ingestion_config.class_weight}")
    
    logger.info(f"<<<<<{STAGE_NAME} Successfully Completed>>>>>")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "PREPARING THE MODEL"
try:
    logger.info(f'{"*"*50}')
    logger.info(f"<<<<<{STAGE_NAME} Started>>>>>")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f"<<<<<{STAGE_NAME} Successfully Completed>>>>>")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training the Model"
try:
    logger.info(f'{"*"*50}')
    logger.info(f"<<<<<{STAGE_NAME} Started>>>>>")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f"<<<<<{STAGE_NAME} Successfully Completed>>>>>")
except Exception as e:
    logger.exception(e)
    raise e
