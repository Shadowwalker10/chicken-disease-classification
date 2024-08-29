from cnnclassifier.logging import logger
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnclassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"<<<<<{STAGE_NAME} Started>>>>>")
    obj = DataIngestionPipeline()
    obj.main()
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
