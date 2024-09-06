from dataclasses import dataclass
from pathlib import Path

## Data Ingestion Entity
@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path
    unzip_dir: Path
    class_weight: Path

## Base Model Preparation Entity
@dataclass(frozen = True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_input_shape: list
    params_batch_size : int
    params_num_classes : int
    params_weights: str
    params_include_top: bool
    params_learning_rate: float
    params_horizontal_flip: bool
    params_rotation_range: float
    params_zoom_range: float
    params_epochs: int
    params_dropout_rate: float
    params_weight_decay: float
    params_freeze_all: bool
    params_freeze_till: int


## Callbacks Entity

from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen = True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    learning_rate: float    


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_input_shape : list
    params_batch_size : int
    params_learning_rate: float
    params_horizontal_flip: bool
    params_rotation_range: float
    params_zoom_range: float
    params_epochs: int
    params_dropout_rate: float
    params_weight_decay: float


@dataclass(frozen = True)
class ModelEvaluationConfig:
    model_path: Path
    evaluation_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int

    