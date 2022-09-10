from Utils.utils import load_config_file


config_file = load_config_file()
STRING_FEATURE = config_file["train_args"]["STRING_FEATURE"]
FLOAT_FEATURE = config_file["train_args"]["FLOAT_FEATURE"]
LABEL = config_file["train_args"]["label_key"]
