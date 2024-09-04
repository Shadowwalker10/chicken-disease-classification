from cnnclassifier.components.model_training import ModelTraining
from cnnclassifier.entity import ModelTrainingConfig
from cnnclassifier.components.prepare_callbacks import PrepareCallbacks
from cnnclassifier.logging import logger
from cnnclassifier.config.configuration import ConfigurationManager
import os

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_model_trainer_config()
        trained_model_path = training_config.trained_model_path
        if os.path.exists(trained_model_path) and os.path.isfile(trained_model_path):
            ## Prompt user to decide whether to train the model
            user_input = input("Do you want to retrain the model? (yes/no): ").strip().lower()
            if user_input in ["yes", "y"]:
                start_training = True
            else:
                start_training = False
                logger.info("Model Training was skipped by the user")
        else:
            logger.info("No trained model found. Starting Training")
            start_training = True

        if start_training:

            config = ConfigurationManager()
            prepare_callbacks_config = config.get_callbacks_config()
            prepare_callbacks = PrepareCallbacks(config = prepare_callbacks_config)
            callbacks_lst = prepare_callbacks.get_tb_ckpt_callbacks()

            training_config = config.get_model_trainer_config()
            data_ingestion_config = config.get_data_ingestion_config()
            training = ModelTraining(config=training_config, dataingestionconfig=data_ingestion_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train(callbacks_lst)



