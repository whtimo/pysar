import pandas as pd
import argparse
from pysar.psi import unwrap_along_edges

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Unwrap along the edges.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input PSC results (H5)")
    parser.add_argument("--psc", type=str, required=True,
                        help="Path to the input PSC CSV file")
    parser.add_argument("--ref_point", type=str, required=True,
                        help="Path to the reference point text file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the unwrapped results (H5)")
    # Parse the arguments
    args = parser.parse_args()

    print("Load data")
    params = unwrap_along_edges.load_network_parameters(args.input_path)
    df = pd.read_csv(args.psc)
    # Rename the unnamed first column to 'point_id'
    df = df.rename(columns={df.columns[0]: 'point_id'})
    # Extract coordinates
    ps_points = df[['sample', 'line']].values
    reference_point = unwrap_along_edges.load_reference_point(args.ref_point)

    edges, temporal_coherence, heights, velocities, residuals = unwrap_along_edges.convert_network_parameters(params)

    print("Create Network")
    G = unwrap_along_edges.create_ps_network(ps_points, edges, temporal_coherence)
    print("Find optimal path")
    paths, distances = unwrap_along_edges.find_optimal_paths_to_reference(G, reference_point)
    print("Extract path parameters")
    path_parameters = unwrap_along_edges.extract_path_parameters(G, paths, heights, velocities, residuals)
    print('Save results')
    unwrap_along_edges.save_path_parameters(path_parameters, df, reference_point, args.output_path)


