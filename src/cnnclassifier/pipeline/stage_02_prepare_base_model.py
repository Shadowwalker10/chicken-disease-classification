from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.prepare_base_model import PrepareBaseModel
from cnnclassifier.logging import logger


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        logger.info("Successfully Downloaded and Updated the Model")
