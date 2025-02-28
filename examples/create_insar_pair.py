from pysar.insar import insar_pair
from pysar.sar import slc

if __name__ == "__main__":

    filename_master = '/home/timo/Data/pysar_test/vesuv/TSX_20240817T165058.615_Vesuv_C571_O055_A_R_SM006_SSC/dims_op_oc_dfd2_694723664_2/TSX-1.SAR.L1B/TDX1_SAR__SSC______SM_S_SRA_20240817T165058_20240817T165102/TDX1_SAR__SSC______SM_S_SRA_20240817T165058_20240817T165102.xml'
    filename_slave = '/home/timo/Data/pysar_test/vesuv/TSX_20240806T165057.831_Vesuv_C570_O055_A_R_SM006_SSC/dims_op_oc_dfd2_694723627_2/TSX-1.SAR.L1B/TDX1_SAR__SSC______SM_S_SRA_20240806T165057_20240806T165101/TDX1_SAR__SSC______SM_S_SRA_20240806T165057_20240806T165101.xml'
    filepath_out = '/home/timo/Data/pysar_test/vesuv/pysar'
    slc_master = slc.fromTSX(filename_master, 0)
    slc_slave = slc.fromTSX(filename_slave, 0)

    pair = insar_pair.createInSarPair(slc_master, slc_slave)

    print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')

    pair.save(directory=filepath_out)