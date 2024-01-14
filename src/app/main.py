import logging
import os
from utils.config import read_config
from utils.logger import setup_logging

def main():
    # Read configuration
    config_path = 'config/config.yaml'
    log_config_path = 'config/log_config.yaml'
    config = read_config(config_path)

    # Configure logging
    log_file = 'logs/app.log'
    setup_logging(log_file, log_config_path)

    logging.info("Starting the application.")

    # Your application logic goes here
    model_architecture = config['model']['architecture']
    training_params = config['training']
    device_params = config['device']

    # Access and use hyperparameters in your code
    logging.info("Model Architecture:")
    logging.info(f"Number of Embeddings (n_embd): {model_architecture['n_embd']}")
    logging.info(f"Number of Heads (n_head): {model_architecture['n_head']}")
    logging.info(f"Number of Layers (n_layer): {model_architecture['n_layer']}")
    logging.info(f"Dropout Rate: {model_architecture['dropout']}")

    logging.info("Training Parameters:")
    logging.info(f"Batch Size: {training_params['batch_size']}")
    logging.info(f"Block Size: {training_params['block_size']}")
    logging.info(f"Max Iterations: {training_params['max_iters']}")
    logging.info(f"Evaluation Interval: {training_params['eval_interval']}")
    logging.info(f"Learning Rate: {training_params['learning_rate']}")
    logging.info(f"Evaluation Iterations: {training_params['eval_iters']}")

    logging.info("Device Configuration:")
    logging.info(f"Device Type: {device_params['type']}")

    logging.info("Application finished.")


if __name__ == "__main__":
    main()
