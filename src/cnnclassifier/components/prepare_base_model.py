import os
from tensorflow.keras import regularizers
import tensorflow as tf
from cnnclassifier.entity import PrepareBaseModelConfig
from pathlib import Path
import json


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    
    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(include_top = self.config.params_include_top,
                                                       weights = self.config.params_weights,
                                                       input_shape = self.config.params_input_shape,
                                                       pooling="max"
                                                       )
        

        
        self.save_model(path = self.config.base_model_path, model = self.model)

    ## Static Method is defined when we want to use a function that doesn't rely on the class
    ## We cannot use self when a function is defined as static method
    ## Static Function can be directly run without instantating the class. Use it directly as
    ## PrepareBaseModel._prepare_full_model()

    @staticmethod
    def _prepare_full_model(model, classes:int, freeze_all:bool, freeze_till:int, learning_rate, weight_decay, dropout_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        #batch_norm = tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)(model.output)
        flatten_in = tf.keras.layers.Flatten()(model.output)
        dense1 = tf.keras.layers.Dense(units = 256, activation = "relu", 
                                       kernel_regularizer = regularizers.l2(weight_decay), 
                                       activity_regularizer=regularizers.l1(0.006))(flatten_in)
        dropout1 = tf.keras.layers.Dropout(dropout_rate)(dense1)
        dense2 = tf.keras.layers.Dense(units = 256, activation = "relu", 
                                       kernel_regularizer = regularizers.l2(weight_decay),
                                       activity_regularizer=regularizers.l1(0.006))(dropout1)
        dropout2 = tf.keras.layers.Dropout(dropout_rate)(dense2)
        prediction = tf.keras.layers.Dense(units = classes,
                                           activation = "softmax")(dropout2)
        full_model = tf.keras.models.Model(model.input, 
                                           prediction)
        full_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                           loss = tf.keras.losses.CategoricalCrossentropy(),
                           metrics = ["accuracy", 
                                      tf.keras.metrics.Precision(), 
                                      tf.keras.metrics.Recall()])
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(model = self.model, 
                                                   classes = self.config.params_num_classes, 
                                                   freeze_all = self.config.params_freeze_all, 
                                                   freeze_till = self.config.params_freeze_till, 
                                                   learning_rate = self.config.params_learning_rate,
                                                   weight_decay = self.config.params_weight_decay,
                                                   dropout_rate = self.config.params_dropout_rate)
        self.save_model(path = self.config.updated_base_model_path, 
                        model = self.full_model)
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            ## Save the model
            model.save(filepath=path)
        except Exception as e:
            print("Serialization error:", e)
            model.save_weights(path)



    