import argparse
from pysar.psi import filter_aps

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="APS phase filtering.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input phase residuals")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output path to the directory saving the filtered aps information into")

    parser.add_argument("--spatial_filter_size", type=float, default=200.0, help="Filter size (float)")

    # Parse the arguments
    args = parser.parse_args()

    filter_aps.process_phase_residuals(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        spatial_filter_size=args.spatial_filter_size
    )