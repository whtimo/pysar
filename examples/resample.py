from pysar import slc, metadata, baseline, insar_pair, coregistration, resample, resampled_pair
import pandas as pd
import rasterio
import numpy as np

if __name__ == "__main__":

    file_pair = '/Users/timo/Documents/WuhanEast/pysar/TDX-1_2018-07-24__2018-08-26.pysar.pair.xml'
    shifts_file = '/Users/timo/Documents/WuhanEast/pysar/shifts.csv'
    output_path = '/Users/timo/Documents/WuhanEast/pysar'

    pair = insar_pair.InSarPair(file_pair)
    shifts = pd.read_csv(shifts_file)

    print(f'Baseline: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')
    print(f'Shifts mean: {shifts['shiftX'].to_numpy().mean().round(2)}, {shifts['shiftY'].to_numpy().mean().round(2)}')
    est_dx, est_dy = coregistration.parameter_estimation(shifts, 1)

    print('Read data')
    master = pair.master.slcdata.read()
    slave = pair.slave.slcdata.read()

    print('Start resampling')
    slave_resample = resample.resample_sar_image(slave, master.shape, est_dx, est_dy)
    #slave_resample = np.zeros(master.shape, dtype=np.complex64)

    res_pair = resampled_pair.createResampledPair(pair.master, pair.slave, slave_resample)
    res_pair_fn, master_tiff_fn, slave_tiff_fn = resampled_pair.createFilenames(res_pair, output_path)

    if not master_tiff_fn.exists():
        print(f'Write master tiff to {master_tiff_fn}')
        # Save the resampled image as a complex float TIFF
        with rasterio.open(
                master_tiff_fn,
                'w',
                driver='GTiff',
                height=master.shape[0],
                width=master.shape[1],
                count=1,
                dtype=np.complex64
        ) as dst:
            dst.write(master, 1)

    print(f'Write slave tiff to {slave_tiff_fn}')
    # Save the resampled image as a complex float TIFF
    with rasterio.open(
        slave_tiff_fn,
        'w',
        driver='GTiff',
        height=master.shape[0],
        width=master.shape[1],
        count=1,
        dtype=np.complex64
    ) as dst:
        dst.write(slave_resample, 1)

    res_pair.save(res_pair_fn, master_tiff_fn, slave_tiff_fn)