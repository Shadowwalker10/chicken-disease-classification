import time
from pathlib import Path
from cnnclassifier.entity import PrepareCallbacksConfig
import os
import tensorflow as tf

class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(self.config.tensorboard_root_log_dir,
                                          f"tb_logs_at_{timestamp}")
        return tf.keras.callbacks.TensorBoard(log_dir = tb_running_log_dir)
    
    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath = self.config.checkpoint_model_filepath, 
                                                  save_best_only = True)
    
    @property
    def _create_reduce_lr_callbacks(self):
        return tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", 
                                                    factor = 0.2, 
                                                    patience = 5, 
                                                    min_lr = 0.001)
    
    @property
    def _create_earlystopping_callbacks(self):
        return tf.keras.callbacks.EarlyStopping(monitor = "val_loss", 
                                                patience = 10)
    
    def get_tb_ckpt_callbacks(self):
        return [self._create_tb_callbacks, 
                 self._create_ckpt_callbacks,
                 self._create_reduce_lr_callbacks, 
                 self._create_earlystopping_callbacks]