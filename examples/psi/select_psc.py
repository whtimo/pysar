from pysar.psi import select_psc
import pandas as pd
import argparse

# Example usage:
if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select PSC or PS points.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input amplitude dispersion TIFF file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--threshold", type=float, defualt=0.25, help="Threshold value for processing (float)")
    parser.add_argument("--window_x", type=int, defualt=3, help="Window size in the x direction (integer)")
    parser.add_argument("--window_y", type=int, default=3, help="Window size in the y direction (integer)")

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    input_path = args.input_path
    output_path = args.output_path
    threshold = args.threshold
    window_x = args.window_x
    window_y = args.window_y

    # Call the processing function
    results = select_psc.process_amplitude_dispersion(
        input_path,
        threshold,
        window_x,
        window_y
    )

    # Save to CSV
    results.to_csv(output_path, index=True)  # added indexing
