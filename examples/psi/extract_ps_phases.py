import pandas as pd
from pysar.psi import extract_ps_phases
import argparse


# Example usage:
if __name__ == "__main__":


    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract the phase information for the PS and remove APS.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the CSV file containing PS coordinates")
    parser.add_argument("--interferogram_dir", type=str, required=True,
                        help="Path to the Interferogram")
    parser.add_argument("--aps_dir", type=str, required=True,
                        help="Path to the APS directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")

    # Parse the arguments
    args = parser.parse_args()

    # Extract phases and save to CSV
    result =  extract_ps_phases.extract_ps_phases(args.input_path, args.interferogram_dir, args.aps_dir)
    result.to_csv(args.output_path, index=True)