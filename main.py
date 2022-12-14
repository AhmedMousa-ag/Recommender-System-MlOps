import os
from Utils.utils import create_pipline, load_config_file, write_to_yaml
from tfx import v1 as tfx  # It's a bit tricky, but you have to import tfx from v1

print('TFX version: {}'.format(tfx.__version__))
VERSION = "1"  # Model version

config_file = load_config_file()
PIPLINE_NAME = config_file["pipline"]["name"]
PIPLINE_ROOT = os.path.join(config_file["directories"]["pipline_path"], PIPLINE_NAME)
METADATA_PATH = os.path.join(PIPLINE_ROOT, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(config_file["directories"]["serving_model"], VERSION)

if not os.path.exists(SERVING_MODEL_DIR):
    os.makedirs(SERVING_MODEL_DIR)

TRANSFORM_MODULE = os.path.join(config_file["directories"]["modules"], "transform_module.py")
TRAINER_MODULE = os.path.join(config_file["directories"]["modules"], "trainer_module.py")
TUNER_MODULE = os.path.join(config_file["directories"]["modules"], "tuner_module.py")
DATA_ROOT = config_file["directories"]["prep_data"]
TUN_BEST_PARAM_ROOT = os.path.join(PIPLINE_ROOT, config_file["directories"]["tun_best_param"])

print("Will initiate pipline.......")
pipline = create_pipline(PIPLINE_NAME, PIPLINE_ROOT, DATA_ROOT, SERVING_MODEL_DIR, METADATA_PATH,
                         TRAINER_MODULE,
                         TRANSFORM_MODULE, TUNER_MODULE, first_time_tuning=True,
                         path_to_tuner_best_hyp=TUN_BEST_PARAM_ROOT)
print("Finished creating pipline.......")

print("Will run the pipline........")
tfx.orchestration.LocalDagRunner().run(pipline)
print("Finished running the pipline.......")
