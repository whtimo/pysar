from pathlib import Path
import argparse
from pysar.psi import residual_interpolation

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Interpolating the phase residuals.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input results from the unwrapping along the edges (H5)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output path to the directory saving the aps information into")

    parser.add_argument("--master_image_width", type=int, required=True, help="Width of the master image (integer)")
    parser.add_argument("--master_image_height", type=int, required=True, help="Height of the master image (integer)")


    # Parse the arguments
    args = parser.parse_args()
    logger = residual_interpolation.setup_logging()

    master_image_width = args.master_image_width  # Timo: add fixed size not based on min/max samples from the PSCs
    master_image_height = args.master_image_height

    grid_size = (master_image_width, master_image_height)

    # Set paths
    input_file = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_file}")
    data = residual_interpolation.load_path_parameters(input_file)

    # Prepare point data
    samples, lines, residuals = residual_interpolation.prepare_point_data(data['points'])

    # Process each epoch
    n_epochs = residuals.shape[1]
    logger.info(f"Processing {n_epochs} epochs")

    for epoch in range(n_epochs):
        logger.info(f"Processing epoch {epoch + 1}/{n_epochs}")

        # Get residuals for current epoch
        epoch_residuals = residuals[:, epoch]

        # Interpolate
        interpolated_nn = residual_interpolation.interpolate_phase_residuals_natural_neighbor(lines, samples, epoch_residuals, grid_size)

        # Save result
        residual_interpolation.save_interpolated_grid(
            interpolated_nn, output_dir, epoch
        )

    logger.info("Processing completed")


