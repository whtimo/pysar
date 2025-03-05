import argparse
import pathlib
from pysar.sar import slc
from pysar import tools
from pysar.insar import resampled_pair, resample, coregistration, insar_pair, flat_interferogram, topo_interferogram
import fnmatch
import os
from datetime import datetime

def find_files(directory, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

# Example usage:
if __name__ == "__main__":

    # # Set up argument parsing
    # parser = argparse.ArgumentParser(description="Select PSC or PS points.")
    # parser.add_argument("--input_path", type=str, required=True,
    #                     help="Path to the input amplitude dispersion TIFF file")
    # parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")
    # parser.add_argument("--threshold", type=float, defualt=0.25, help="Threshold value for processing (float)")
    # parser.add_argument("--window_x", type=int, defualt=3, help="Window size in the x direction (integer)")
    # parser.add_argument("--window_y", type=int, default=3, help="Window size in the y direction (integer)")
    #
    # # Parse the arguments
    # args = parser.parse_args()

    root_dir = pathlib.Path('/home/timo/Data/LasVegasDesc_pysar_psi')
    swath_id = 0
    srtm_path = '/home/timo/Data/LasVegasDesc_pysar_psi/srtm_13_05.tif'

    data_dir = root_dir / 'data'
    if not data_dir.exists():
        print('Data directory does not exist')
        exit(1)

    slc_dir = root_dir / 'imported'
    slc_dir.mkdir(parents=True, exist_ok=True)
    resampled_dir = root_dir / 'resampled'
    resampled_dir.mkdir(parents=True, exist_ok=True)
    flat_dir = root_dir / 'flat'
    flat_dir.mkdir(parents=True, exist_ok=True)
    topo_dir = root_dir / 'topo'
    topo_dir.mkdir(parents=True, exist_ok=True)

    tsx_files = find_files(data_dir, 'T*_SAR__SSC_*.xml')
    for tsx_file in tsx_files:
        sc = slc.fromTSX(tsx_file, swath_id)
        sc.save(directory=slc_dir)

    slc_files = find_files(slc_dir, '*slc.xml')
    slc_dates = []
    for slc_file in slc_files:
        date_str = slc_file.split('_')[-1].split('.')[0]  # Extracts '2010-08-11'
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        slc_dates.append((date_obj, slc_file))

    sorted_slc_dates = sorted(slc_dates, key=lambda x: x[0])
    master_ix = int(len(sorted_slc_dates) / 2)

    slc_master = slc.fromPysarXml(sorted_slc_dates[master_ix][1])
    for ix in range(len(sorted_slc_dates)):
        if ix != master_ix:
            slc_slave = slc.fromPysarXml(sorted_slc_dates[ix][1])
            pair = insar_pair.createInSarPair(slc_master, slc_slave)
            print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')
            pair.save(directory=slc_dir)

    pair_files = find_files(slc_dir, '*pysar.pair.xml')
    for pair_file in pair_files:
        pair = insar_pair.InSarPair(pair_file)
        date_str = pair_file.split('_')[-1].split('.')[0]
        shifts = coregistration.subpixel_shifts(pair.master.metadata, pair.slave.metadata, int(pair.shift_x),
                                                int(pair.shift_y), pair.master.slcdata.read(),
                                                pair.slave.slcdata.read(), output=tools.output_console, points=3200)
        print(
            f'Baseline: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')
        print(
            f'Shifts mean: {shifts['shiftX'].to_numpy().mean().round(2)}, {shifts['shiftY'].to_numpy().mean().round(2)}')

        est_dx, est_dy = coregistration.parameter_estimation(shifts, 2)

        print('Read data')
        master = pair.master.slcdata.read()
        slave = pair.slave.slcdata.read()

        print('Start resampling')
        slave_resample = resample.resample_sar_image(slave, master.shape, est_dx, est_dy, output=tools.output_console)
        # slave_resample = np.zeros(master.shape, dtype=np.complex64)

        res_pair = resampled_pair.createResampledPair(pair.master, pair.slave, slave_resample)
        master_tiff_filename = res_pair.save(directory=resampled_dir, overwrite=False)

resampled_files = find_files(resampled_dir, '*pysar.resampled.xml')
for resampled_file in resampled_files:
    pair = resampled_pair.ResampledPair(resampled_file)
    phase_model, poly = flat_interferogram.get_flat_phase_model(pair)

    flat_phase = flat_interferogram.create_flattened_interferogram(pair, phase_model, poly, tools.output_console)

    flatty = flat_interferogram.createFlatInterferogram(pair.master.metadata, pair.slave.metadata, flat_phase)
    flatty.save(directory=topo_dir)

interfero_files = find_files(flat_dir, '*pysar.flat.interfero.xml')
for interfero_file in interfero_files:
    interfero = flat_interferogram.FlatInterferogram(interfero_file)

    topo_phases = topo_interferogram.get_topographic_phases(interfero.master_metadata,
                                                     interfero.slave_metadata,
                                                     srtm_path)

    flat_phases = interfero.read()
    rem_phases = topo_interferogram.remove_topographic_phases(flat_phases, topo_phases)

    toppy = topo_interferogram.createTopoInterferogram(interfero.master_metadata, interfero.slave_metadata, rem_phases)
    toppy.save(directory=topo_dir)