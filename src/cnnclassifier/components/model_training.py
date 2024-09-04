from cnnclassifier.entity import ModelTrainingConfig, DataIngestionConfig
import tensorflow as tf
from pathlib import Path
import json


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig, dataingestionconfig: DataIngestionConfig):
        self.config = config
        self.data_ingestion_config = dataingestionconfig

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        valid_datagenerator_kwargs = dict(rescale = 1./255, 
                                    validation_split = 0.20)
        
        train_datagenerator_kwargs = dict(rescale = 1./255, 
                            validation_split = 0.20,
                            horizontal_flip = self.config.params_horizontal_flip,
                            rotation_range = self.config.params_rotation_range,
                            zoom_range = self.config.params_zoom_range)
        
        dataflow_kwargs = dict(target_size = self.config.params_input_shape[:-1], 
                               batch_size = self.config.params_batch_size,
                               interpolation = "bilinear")
        
        #** symbol is used for unpacking dictionaries into keyword arguments.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**valid_datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(self.config.training_data, 
                                                                       subset = "validation", 
                                                                       shuffle = True, 
                                                                       **dataflow_kwargs)
        
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**train_datagenerator_kwargs)
        self.train_generator = train_datagenerator.flow_from_directory(self.config.training_data,
                                                                       subset = "training", 
                                                                       shuffle = True, 
                                                                       **dataflow_kwargs)
        
    @staticmethod
    def save_model(path: Path , model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        with open(self.data_ingestion_config.class_weight, "r") as f:
            saved_weights = json.load(f)
        class_weight = {int(k): v for k,v in saved_weights.items()}


        self.model.fit(self.train_generator, 
                       epochs = self.config.params_epochs, 
                       steps_per_epoch = self.steps_per_epoch, 
                       validation_steps = self.validation_steps,
                       callbacks = callback_list, 
                       validation_data = self.valid_generator,
                       class_weight = class_weight
                       )
        
        self.save_model(path = self.config.trained_model_path,
                         model = self.model)
        
