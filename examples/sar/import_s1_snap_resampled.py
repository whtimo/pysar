from pysar.sar import slc, metadata
from pysar.insar import resampled_pair

if __name__ == "__main__":

    file_name = '/home/timo/Data/LVS1_pysar/deburst/S1A_IW_SLC__1SDV_20230208T134359_20230208T134427_047145_05A808_56A7_Orb_Stack_esd_deb.dim'
    output_path = '/home/timo/Data/LVS1_pysar'

    pairs = resampled_pair.fromDimS1Deburst(file_name)

    for pair in pairs:
        pair.save(directory=output_path, overwrite=False)