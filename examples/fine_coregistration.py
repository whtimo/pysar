from pysar import slc, metadata, baseline, insar_pair, coregistration
import pandas as pd

if __name__ == "__main__":

    file_pair = '/Users/timo/Documents/WuhanEast/pysar/TDX-1_2018-07-24__2018-08-26.pysar.pair.xml'
    shifts_file = '/Users/timo/Documents/WuhanEast/pysar/shifts.csv'

    pair = insar_pair.InSarPair(file_pair)

    print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')

    shifts = coregistration.subpixel_shifts(pair.master.metadata, pair.slave.metadata, int(pair.shift_x), int(pair.shift_y), pair.master.slcdata.read(), pair.slave.slcdata.read())
    shifts.to_csv(shifts_file, index=True)





