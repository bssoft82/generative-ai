import logging

import yaml

class Config:
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.n_embd = config['model']['architecture']['n_embd']
        self.n_head = config['model']['architecture']['n_head']
        self.n_layer = config['model']['architecture']['n_layer']
        self.dropout = config['model']['architecture']['dropout']

        self.batch_size = config['training']['batch_size']
        self.block_size = config['training']['block_size']
        self.max_iters = config['training']['max_iters']
        self.eval_interval = config['training']['eval_interval']
        self.learning_rate = float(config['training']['learning_rate'])
        self.eval_iters = config['training']['eval_iters']

        self.device = config['device']['type']


    def log_state(cls):
        """
        Log the current state of the Config class.
        """
        logging.info("Config State:")
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                logging.info(f"{key}: {value}")