from pysar import tools
from pysar.insar import resampled_pair, resample, coregistration, insar_pair
import pandas as pd

if __name__ == "__main__":

    file_pair = '/home/timo/Data/pysar_test/lasvegas_desc/pysar/TSX-1_0_2010-08-28__TSX-1_2010-06-23.pysar.pair.xml'
    shifts_file = '/home/timo/Data/pysar_test/lasvegas_desc/pysar/TSX-1_2010-06-23-9_shifts.csv'
    output_path = '/home/timo/Data/pysar_test/lasvegas_desc/pysar'
    #file_pair = '/home/timo/Data/pysar_test/vesuv/pysar/TDX-1_0_2024-08-17__TDX-1_2024-08-06.pysar.pair.xml'
    #shifts_file = '/home/timo/Data/pysar_test/vesuv/pysar/TDX-1_2024-08-06_shifts.csv'
    #output_path = '/home/timo/Data/pysar_test/vesuv/pysar'

    pair = insar_pair.InSarPair(file_pair)
    shifts = pd.read_csv(shifts_file)

    print(f'Baseline: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')
    print(f'Shifts mean: {shifts['shiftX'].to_numpy().mean().round(2)}, {shifts['shiftY'].to_numpy().mean().round(2)}')


    est_dx, est_dy = coregistration.parameter_estimation(shifts, 2)

    print('Read data')
    master = pair.master.slcdata.read()
    slave = pair.slave.slcdata.read()

    print('Start resampling')
    slave_resample = resample.resample_sar_image(slave, master.shape, est_dx, est_dy, output=tools.output_console)
    #slave_resample = np.zeros(master.shape, dtype=np.complex64)

    res_pair = resampled_pair.createResampledPair(pair.master, pair.slave, slave_resample)
    master_tiff_filename = res_pair.save(directory=output_path)