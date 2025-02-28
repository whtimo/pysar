from pysar.insar import insar_pair
from pysar.sar import slc

if __name__ == "__main__":

    filename_master = '/home/timo/Data/pysar_test/lasvegas_desc/data/dims_op_oc_dfd2_369802346_3/TSX-1.SAR.L1B/TSX1_SAR__SSC______HS_S_SRA_20100828T134953_20100828T134954/TSX1_SAR__SSC______HS_S_SRA_20100828T134953_20100828T134954.xml'
    filename_slave = '/home/timo/Data/pysar_test/lasvegas_desc/data/dims_op_oc_dfd2_369802346_3/TSX-1.SAR.L1B/TSX1_SAR__SSC______HS_S_SRA_20100623T134949_20100623T134950/TSX1_SAR__SSC______HS_S_SRA_20100623T134949_20100623T134950.xml'
    filepath_out = '/home/timo/Data/pysar_test/lasvegas_desc/pysar'
    slc_master = slc.fromTSX(filename_master, 0)
    slc_slave = slc.fromTSX(filename_slave, 0)

    pair = insar_pair.createInSarPair(slc_master, slc_slave)

    print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')

    pair.save(directory=filepath_out)