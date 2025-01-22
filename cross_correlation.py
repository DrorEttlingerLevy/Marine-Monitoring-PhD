import sys
sys.path.append(r'C:\Users\dror.e\Documents')

import functions as func


import numpy as np
import pickle
from multiprocessing import Pool
import logging
import time

import numpy as np
import pickle
from multiprocessing import Pool, log_to_stderr
import logging
import time

# Configure logging for both the main process and multiprocessing workers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_to_stderr().setLevel(logging.INFO)



if __name__ == "__main__":
    # Example usage:
    template_path = 'segment_675_5_676_5.txt'
    pkl_file_path = r"C:\Users\dror.e\test_folder\241030-T007.pkl"
    channel_indices = slice(0, 6)

    logging.info("Starting cross-correlation computation...")
    results = func.perform_cross_correlation_for_channels(template_path, pkl_file_path, channel_indices)
    logging.info("Cross-correlation computation completed.")