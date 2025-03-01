import numpy as np
#from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Union
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
#import h5py
import time
from multiprocessing import Pool
import argparse
from pysar.psi import ps_parameter_estimation_periodogram_parallel


if __name__ == "__main__":
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Estimate parameters")
    parser.add_argument("--ps_phases", type=str, required=True,
                        help="Path to the PS phases CSV file")
    parser.add_argument("--psc", type=str, required=True,
                        help="Path to the PSC CSV file")
    parser.add_argument("--ps", type=str, required=True,
                        help="Path to the PS CSV file")
    parser.add_argument("--ref_point", type=str, required=True,
                        help="Path to the reference point text file")
    parser.add_argument("--interferogram_dir", type=str, required=True,
                        help="Path to the Interferogram")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")

    # Parse the arguments
    args = parser.parse_args()

    # Read the CSV file
    df_psc = pd.read_csv(args.psc)
    df_ps = pd.read_csv(args.ps_phases)
    # Get the column names that are dates (skip the first 3 columns)
    date_columns = df_ps.columns[3:]

    # Convert the date strings to datetime objects and store in a list
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in date_columns]

    #ref_point = 0
    ref_point = ps_parameter_estimation_periodogram_parallel.find_matching_point_index(args.ref_point, args.psc, args.ps)
    #print("Reading the network") # Adding some comments because it is a long process
    #ps_network = PSNetwork(dates, "/path/to/xml/files")
    ps_info = ps_parameter_estimation_periodogram_parallel.PSInfo(dates, args.interferogram_dir,  args.ps_phases)

    parameter_estimator = ps_parameter_estimation_periodogram_parallel.ParameterEstimator(ps_info)
    print("Start parameter estimation") # Adding some comments because it is a long process
    params = parameter_estimator.estimate_parameters(ref_point)
    print("Save parameters") # Adding some comments because it is a long process
    #save_network_parameters(params, ps_network, '/home/timo/Data/LasVegasDesc/ps_results3_perio_year.h5')
    ps_parameter_estimation_periodogram_parallel.save_point_data_to_csv(args.ps_phases, args.output_path, params)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


