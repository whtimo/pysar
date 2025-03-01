import logging
import argparse
from pysar.psi import amplitude_dispersion

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate amplitude dispersion from SAR images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing complex SAR images")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path for the amplitude dispersion TIFF")

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    input_dir = args.input_dir
    output_file = args.output_file
    try:
        stats = amplitude_dispersion.calculate_amplitude_dispersion(input_dir, output_file)
        print("Statistics:", stats)
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")