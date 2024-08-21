import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "cnnclassifier"

lst_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params/params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"
]

directories = {str(Path(p).parent) for p in lst_of_files}
[os.makedirs(dir, exist_ok = True) for dir in directories]
logging.info("Parent Directories Created.")

## Creating files inside individual directories
[Path(p).touch(exist_ok=True) for p in lst_of_files]
logging.info("Individual Directories Created.")

