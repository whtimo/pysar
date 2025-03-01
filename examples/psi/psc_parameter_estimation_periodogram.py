import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Union
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import argparse
from pysar.psi import psc_parameter_estimation_periodogram

# Example usage:
if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Estimate parameters along the network.")
    parser.add_argument("--psc_phases", type=str, required=True,
                        help="Path to the PSC phases CSV file")
    parser.add_argument("--interferogram_dir", type=str, required=True,
                        help="Path to the Interferogram")
    parser.add_argument("--triangles", type=str, required=True,
                        help="Path to the Triangulation CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output H5 file")

    # Parse the arguments
    args = parser.parse_args()

    # Read the CSV file
    #df = pd.read_csv('your_file.csv')
    df = pd.read_csv(args.psc_phases)

    # Get the column names that are dates (skip the first 3 columns)
    date_columns = df.columns[3:]

    # Convert the date strings to datetime objects and store in a list
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in date_columns]

    print("Reading the network") # Adding some comments because it is a long process
    ps_network = psc_parameter_estimation_periodogram.PSNetwork(dates, args.interferogram_dir, args.triangles, args.psc_phases)

    parameter_estimator = psc_parameter_estimation_periodogram.NetworkParameterEstimator(ps_network)
    print("Start parameter estimation") # Adding some comments because it is a long process
    params = parameter_estimator.estimate_network_parameters()
    print("Save parameters") # Adding some comments because it is a long process
    psc_parameter_estimation_periodogram.save_network_parameters(params, ps_network, args.output_path)



