import yaml
import os
from Utils.utils import create_pipline
VERSION = "1"

with open("config.yaml", "r") as fw:
    config_file = yaml.safe_load(fw)
PIPLINE_NAME = config_file["pipline"]["name"]
PIPLINE_ROOT = os.path.join(config_file["directories"]["pipline_path"], PIPLINE_NAME)
METADATA_PATH = os.path.join(PIPLINE_ROOT, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(config_file["directories"]["serving_model"], VERSION)
TRANSFORM_MODULE = os.path.join(config_file["directories"]["modules"],"transform_module.py")
TRAINER_MODULE = os.path.join(config_file["directories"]["modules"],"trainer_module.py")


