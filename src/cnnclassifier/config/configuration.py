from cnnclassifier.constants import *
from cnnclassifier.utils.common import *
from cnnclassifier.entity import *
import os
from pathlib import Path

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH, 
                 params_filepath = PARAMS_FILE_PATH,
                 secret_filepath = SECRET_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.secret = read_yaml(secret_filepath)
        create_directories_files([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories_files([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            dataset_name = config.dataset_name,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir,
            class_weight = config.class_weight
            )
        return data_ingestion_config

    def get_prepare_base_model_config(self)->PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories_files([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(root_dir = Path(config.root_dir), 
                                                           base_model_path = Path(config.base_model_path),
                                                           updated_base_model_path = Path(config.updated_base_model_path),
                                                           params_input_shape = self.params.input_shape,
                                                           params_batch_size = self.params.batch_size,
                                                           params_num_classes = self.params.num_classes,
                                                           params_weights = self.params.weights,
                                                           params_include_top = self.params.include_top,
                                                           params_learning_rate = self.params.learning_rate,
                                                           params_horizontal_flip = self.params.horizontal_flip,
                                                           params_rotation_range = self.params.rotation_range,
                                                           params_zoom_range = self.params.zoom_range,
                                                           params_epochs = self.params.epochs,
                                                           params_dropout_rate = self.params.dropout_rate,
                                                           params_weight_decay = self.params.weight_decay,
                                                           params_freeze_all = self.params.freeze_all,
                                                           params_freeze_till = self.params.freeze_till
                                                           )
        
        return prepare_base_model_config
    
    def get_callbacks_config(self)-> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        create_directories_files([config.root_dir])
        create_directories_files([config.tensorboard_root_log_dir])
        create_directories_files([os.path.dirname(config.checkpoint_model_filepath)])


        prepare_callbacks_config = PrepareCallbacksConfig(root_dir = Path(config.root_dir), 
                                                          tensorboard_root_log_dir= Path(config.tensorboard_root_log_dir),
                                                          checkpoint_model_filepath = Path(config.checkpoint_model_filepath),
                                                          learning_rate = self.params.learning_rate)
        
        return prepare_callbacks_config
    
    def get_model_trainer_config(self) -> ModelTrainingConfig:
        model_training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(self.config.data_ingestion.local_data_file, "Train")
        create_directories_files([Path(model_training.root_dir)])


        model_training_config = ModelTrainingConfig(root_dir = Path(model_training.root_dir),
                                                    trained_model_path = Path(model_training.trained_model_path),
                                                    updated_base_model_path = Path(prepare_base_model.updated_base_model_path),
                                                    training_data = training_data,
                                                    params_input_shape = self.params.input_shape,
                                                    params_batch_size = self.params.batch_size,
                                                    params_learning_rate = self.params.learning_rate, 
                                                    params_horizontal_flip = self.params.horizontal_flip,
                                                    params_rotation_range = self.params.rotation_range,
                                                    params_zoom_range = self.params.zoom_range,
                                                    params_epochs = self.params.epochs, 
                                                    params_dropout_rate = self.params.dropout_rate,
                                                    params_weight_decay = self.params.weight_decay)
        return model_training_config
    

    def get_evaluation_config(self)-> ModelEvaluationConfig:
        config = self.config.model_evaluation

        model_eval_config = ModelEvaluationConfig(model_path=config.model_path, 
                                                    evaluation_data = config.evaluation_data,
                                                    all_params= self.params,
                                                    params_image_size = self.params.input_shape,
                                                    params_batch_size = self.params.batch_size
                                                    )
        
        return model_eval_config