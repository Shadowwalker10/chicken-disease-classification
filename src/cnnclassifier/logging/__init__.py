import os
import sys
import logging
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s : %(module)s : %(message)s]"
logfile_path = "./logs/running_logs.log"

os.makedirs(name = str(Path(logfile_path).parent), exist_ok = True)
Path(logfile_path).touch(exist_ok = True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers = [
        logging.FileHandler(logfile_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cnnlogger")