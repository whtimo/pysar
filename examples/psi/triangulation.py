import pandas as pd
from pysar.psi import triangulation
import argparse

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Triangulation the PSC.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input PSC file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output CSV file")

    # Parse the arguments
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    results_df = triangulation.triangulate_psc(df)
    # Save to CSV
    #results_df.to_csv('triangulation_results.csv', index=False)
    results_df.to_csv(args.output_path, index=False)