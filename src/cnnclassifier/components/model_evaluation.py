## Update the components
from cnnclassifier.entity import ModelEvaluationConfig
from pathlib import Path
import tensorflow as tf
from cnnclassifier.utils.common import save_json
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def _valid_generator(self):
        datagenerator_kwargs = dict(rescale = 1./255, 
                                    validation_split = 0.3)
        
        dataflow_kwargs = dict(target_size = self.config.params_image_size[:-1], 
                               batch_size = self.config.params_batch_size,
                               interpolation = "bilinear")
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self._valid_generator = valid_datagenerator.flow_from_directory(directory = self.config.evaluation_data, 
                                                                        subset = "validation", 
                                                                        shuffle = True, 
                                                                        **dataflow_kwargs)
        
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self._valid_generator()
        self.score = self.model.evaluate(self._valid_generator)

    def save_score(self):
        loss, accuracy, precision, recall = self.score[0], self.score[1], self.score[2], self.score[3]
        scores = {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall}
        save_json(path = Path("scores.json"), data = scores)

    def display_images_with_prediction(self, num_images = 10, title_fontsize = 8):
        images, labels = next(self._valid_generator)
        num_images = min(num_images, len(images))
        predictions = self.model.predict(images)

        predicted_labels = np.argmax(predictions, axis = 1)
        original_labels = np.argmax(labels, axis = 1)

        ##Extracting class labels
        class_labels = list(self._valid_generator.class_indices.keys())

        plt.figure(figsize = (10,10))
        for i in range(num_images):
            plt.subplot(5,5,i+1)
            plt.imshow(images[i])
            plt.title(f"True: {class_labels[original_labels[i]]}\nPred: {class_labels[predicted_labels[i]]}", 
                      fontsize = title_fontsize)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        

        


