import logging
import glob

def setup_logging():
    # Configure logging

    log_files = glob.glob('adverserial-attacks/logs/*.log')

    # create log file name with incremental version
    log_file_name = 'adverserial-attacks/logs/applogs_'
    if log_files:
        log_file_name += str(len(log_files) + 1)
    else:
        log_file_name += str(1)
    log_file_name += '.log'
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
