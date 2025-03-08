from pysar.insar import insar_pair
from pysar.sar import slc

if __name__ == "__main__":

    filename_master = ''
    filename_slave = ''
    filepath_out = ''
    slc_master = slc.fromTSX(filename_master, 0)
    slc_slave = slc.fromTSX(filename_slave, 0)

    pair = insar_pair.createInSarPair(slc_master, slc_slave)

    print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')

    pair.save(directory=filepath_out)