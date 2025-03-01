import numpy as np
import h5py
import argparse
from pysar.psi import select_reference_point

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select the reference point.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input results from the estimation along the edges (H5)")
    parser.add_argument("--ref_point", type=str, required=True,
                        help="Output path to the reference point text file")

    parser.add_argument("--threshold", type=float, defualt=0.8, help="Temporal Coherence Threshold (float)")
    parser.add_argument("--min_connections", type=int, defualt=7, help="Min number of connections (integer)")

    # Parse the arguments
    args = parser.parse_args()

    #params = load_network_parameters('filename')
    params = select_reference_point.load_network_parameters(args.input_path)
    reference_point = select_reference_point.select_reference_point(params, temporal_coherence_threshold=args.threshold, min_connections=args.min_connections)
    select_reference_point.save_reference_point(reference_point, args.ref_point)
