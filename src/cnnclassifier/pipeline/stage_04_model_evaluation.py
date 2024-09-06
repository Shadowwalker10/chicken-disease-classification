from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.model_evaluation import Evaluation

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config = model_eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.display_images_with_prediction()
        