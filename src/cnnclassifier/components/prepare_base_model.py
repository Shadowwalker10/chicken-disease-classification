import os
import tensorflow as tf
from cnnclassifier.entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(include_top = self.config.params_include_top,
                                                       weights = self.config.params_weights,
                                                       input_shape = self.config.params_input_shape
                                                       )
        
        self.save_model(path = self.config.base_model_path, model = self.model)

    ## Static Method is defined when we want to use a function that doesn't rely on the class
    ## We cannot use self when a function is defined as static method
    ## Static Function can be directly run without instantating the class. Use it directly as
    ## PrepareBaseModel._prepare_full_model()

    @staticmethod
    def _prepare_full_model(model, classes:int, freeze_all:bool, freeze_till:int, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        flatten_in = tf.keras.layers.Flatten()(model.output)
        dense1 = tf.keras.layers.Dense(units = 2048, activation = "relu")(flatten_in)
        dense2 = tf.keras.layers.Dense(units = 128, activation = "relu")(dense1)
        prediction = tf.keras.layers.Dense(units = classes,
                                           activation = "softmax")(dense2)
        full_model = tf.keras.models.Model(model.input, 
                                           prediction)
        full_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                           loss = tf.keras.losses.CategoricalCrossentropy(),
                           metrics = ["accuracy", "precision", "recall"])
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(model = self.model, 
                                                   classes = self.config.params_num_classes, 
                                                   freeze_all = self.config.params_freeze_all, 
                                                   freeze_till = self.config.params_freeze_till, 
                                                   learning_rate = self.config.params_learning_rate)
        self.save_model(path = self.config.updated_base_model_path, 
                        model = self.full_model)
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(filepath = path)



    