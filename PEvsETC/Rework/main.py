import sys
sys.path.append("./src")

from complexity import *
from simulations import *

import h5py
import logging
import time

import numpy as np

def setup_logging(log_filename="experiment_run.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_filename, mode='w')    

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    logging.info("Logging initialized. Writing to console and file.")


def run_experiments(filename, a_values, L, D_array, bin_array, transient, t, noise_values):
    logging.info("="*40)
    logging.info(f"STARTING EXPERIMENT")
    logging.info(f"Target File: {filename}")
    logging.info("="*40)

    try:
        with h5py.File(filename, 'w') as f:
            f.attrs['description'] = "Logistic Map Complexity Comparision"
            f.attrs['created_on'] = time.ctime()
            f.attrs['seq_length'] = L

            logging.info("Pre-calculating Lyapunov exponents (Noise Free)...")
            lyap_arr = np.zeros(len(a_values))
            for i, a in enumerate(a_values):
                lyap_arr[i] = lyapunov_exponent(a, L, transient)

            for noise_idx, noise in enumerate(noise_values):
                logging.info(f"[{noise_idx+1}/{len(noise_values)}] Processing Noise Level Sigma = {noise}")
                start_time_noise = time.time()

                n_group = f.create_group(f"noise_{noise}")
                n_group.attrs['noise'] = noise

                pe_results = {D: np.zeros(len(a_values)) for D in D_array}
                etc_results = {bins: np.zeros(len(a_values)) for bins in bin_array}


                log_interval = max(1, len(a_values) //5)

                for i, a in enumerate(a_values):
                    if i % log_interval == 0:
                        logging.info(f"     ... Simulating a = {a:.4f} ({i}/{len(a_values)})")

                    series = simulate(a, L, transient, noise)

                    for D in D_array:
                        pe_results[D][i] = pe_method(series, D, t)
                    
                    for bins in bin_array:
                        etc_results[bins][i] = etc_method(series, bins)

                logging.info(f"     Saving Data for noise={noise} to HDF5...")
                n_group.create_dataset("a_values", data=a_values)
                n_group.create_dataset("lyapunov", data=lyap_arr)

                pe_group = n_group.create_group("permutation_entropy")
                for D, data in pe_results.items():
                    dset = pe_group.create_dataset(f"D_{D}", data = data)
                    dset.attrs['embedding_dimension'] = D

                etc_group = n_group.create_group("etc")
                for bins, data in etc_results.items():
                    dset = etc_group.create_dataset(f"bins_{bins}", data=data)
                    dset.attrs['num_bins'] = bins

                elapsed = time.time() - start_time_noise
                logging.info(f"     Completed noise={noise} in {elapsed:.2f} seconds")

        logging.info("="*40)
        logging.info(f"Data saved to {filename}")
   
    except Exception as e:
        logging.error("Failure", exc_info=True)
        raise e






if __name__=="__main__":
    #L_array = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    L_array = [100000, 1000000, 10000000, 100000000]
    transient = 1000
    #D_arrays = [[3], [3,5], [3,5], [3,5,7], [3,5,7], [3,5,7,9], [3,5,7,9]]
    D_arrays = [[3,5,7], [3,5,7], [3,5,7,9], [3,5,7,9]]
    t = 1
    bin_array = [2,3,4,5]
    noise_values = [0.0, 0.01, 0.02, 0.05, 0.1]

    a_values = np.linspace(3.5, 4, 500)
    

    setup_logging("simulation_log.txt")

    for i, L in enumerate(L_array):
        filename = f"logistic_map_{L}.h5"
        D_array = D_arrays[i]
        
        run_experiments(
                filename, 
                a_values, 
                L, 
                D_array, 
                bin_array, 
                transient, 
                t, 
                noise_values)
