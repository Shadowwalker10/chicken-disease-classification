import numpy as np
import os
import tensorflow as tf
from cnnclassifier.config.configuration import ConfigurationManager
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class PredictionPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        

    def predict(self):
        ## Load the model
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_path = model_trainer_config.trained_model_path
        image_size = model_trainer_config.params_input_shape
        model = tf.keras.models.load_model(model_path)


        img = image.load_img(path=self.filepath,
                             target_size = image_size[:2])
        img_array = image.img_to_array(img)
        img_array = img_array/255.0 #normalize the image
        img_array_expanded = np.expand_dims(img_array, axis = 0) #expand to add batch dimension
        

        ## Predictions
        predictions_probs = model.predict(img_array_expanded)
        conf= np.max(predictions_probs)
        predictions = np.argmax(predictions_probs,
                                axis = 1)
        classes = ["cocci", 
                   "healthy", 
                   "ncd", 
                   "pcrcocci",
                   "pcrhealthy",
                   "pcrncd", 
                   "pcrsalmo",
                   "salmo"]
        prediction = str.upper(classes[predictions[0]])
        confidence = np.round(conf*100, 2)
        
        
  
        return prediction, confidence

        # plt.figure(figsize = (5,5))
        # plt.imshow(img_array)
        # plt.title(f"Predicted Class: {prediction}\nConfidence: {confidence: .2f}", 
        #           fontsize = 8, 
        #           fontcolor = "red")

        





        



