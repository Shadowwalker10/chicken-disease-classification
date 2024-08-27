from cnnclassifier.logging import logger
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"<<<<<{STAGE_NAME} Started>>>>>")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f"<<<<<{STAGE_NAME} Successfully Completed>>>>>")
except Exception as e:
    logger.exception(e)
    raise e
