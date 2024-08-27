import os
from box.exceptions import BoxValueError
import yaml
import json
from cnnclassifier.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Union, List
import base64
import zipfile

@ensure_annotations
def read_yaml(path_to_yaml_file: Path)->ConfigBox:
    try:
        with open(path_to_yaml_file, "r") as file:
            content = yaml.safe_load(file)
            logger.info(f"{path_to_yaml_file} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file empty")
    except Exception as e:
        logger.exception(e)
        raise e
    
@ensure_annotations
def create_directories_files(lst_path: List, verbose = True):
    directories = [str(Path(p)) for p in lst_path]
    [os.makedirs(p, exist_ok = True) for p in directories]

    ##Creating files inside directories
    [Path(p).touch(exist_ok = True) for p in lst_path]
    if verbose:
        logger.info("Parent Directories and Files Successfully Created")

@ensure_annotations
def save_json(path: Path, data:dict):
    with open(path, "w") as file:
        json.dump(data, file, indent = 4)
    logger.info(f"Json File saved to path: {path}")

@ensure_annotations
def load_json(path: Path, verbose = True):
    with open(path, "r") as f:
        content = json.load(f)
    if verbose:logger.info(f"{path} loaded successfully")
    return ConfigBox(content)

@ensure_annotations
def decodeImage(Imgstring, filename):
    imgdata = base64.b64decode(Imgstring)
    with open(filename, "wb") as f:
        f.write(imgdata)
       
    logger.info(f"{filename} updated successfully")

@ensure_annotations
def encode_image_b64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    
@ensure_annotations
def extract_zipfile(zipfile_path:Path, output_path: Path):
    logger.info(f"Extracting {zipfile_path} to {output_path}")
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    logger.info("Files Extracted Successfully")

@ensure_annotations
def get_size(path: Path) -> int:
    try:
        size = os.path.getsize(path)
        return round(size / 1024)  # Return size in kilobytes
    except FileNotFoundError:
        #logger.error(f"File not found: {path}")
        return 0  # Return 0 if file is not found