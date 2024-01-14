import logging
from src.utils.config import read_config

def setup_logging(log_file, log_config_path):
    config = read_config(log_config_path)

    logging.basicConfig(filename=log_file, level=config['log_level'], format=config['log_format'])
    console = logging.StreamHandler()
    console.setLevel(config['log_level'])
    formatter = logging.Formatter(config['log_format'])
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
