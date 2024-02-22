import logging

import yaml


def setup_logging(log_file, log_config_path):
    with open(log_config_path, 'r') as file:
        log_config = yaml.safe_load(file)

    logging.basicConfig(filename=log_file, level=log_config['log_level'], format=log_config['log_format'])
    console = logging.StreamHandler()
    console.setLevel(log_config['log_level'])
    formatter = logging.Formatter(log_config['log_format'])
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
