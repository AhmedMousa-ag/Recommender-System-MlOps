import yaml
import os
from Utils.utils import create_pipline
from tfx import v1 as tfx  # It's a bit tricky but you have to import tfx from v1

print('TFX version: {}'.format(tfx.__version__))
VERSION = "1"

with open("config.yaml", "r") as fw:
    config_file = yaml.safe_load(fw)
PIPLINE_NAME = config_file["pipline"]["name"]
PIPLINE_ROOT = os.path.join(config_file["directories"]["pipline_path"], PIPLINE_NAME)
METADATA_PATH = os.path.join(PIPLINE_ROOT, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(config_file["directories"]["serving_model"], VERSION)
TRANSFORM_MODULE = os.path.join(config_file["directories"]["modules"], "transform_module.py")
TRAINER_MODULE = os.path.join(config_file["directories"]["modules"], "trainer_module.py")
TUNER_MODULE = os.path.join(config_file["directories"]["modules"], "tuner_module.py")
DATA_ROOT = config_file["directories"]["prep_data"]

print("Will initiate pipline.......")
pipline = create_pipline(PIPLINE_NAME, PIPLINE_ROOT, DATA_ROOT, SERVING_MODEL_DIR, METADATA_PATH, TRAINER_MODULE,
                         TRANSFORM_MODULE, TUNER_MODULE)
print("Finished creating pipline.......")

print("Will run the pipline........")
tfx.orchestration.LocalDagRunner().run(pipline)
print("Finished running the pipline.......")
