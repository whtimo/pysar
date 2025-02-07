from pysar import slc, metadata, baseline, insar_pair

if __name__ == "__main__":

    filename_master = '/Users/timo/Documents/WuhanEast/dims_op_oc_dfd2_675921665_1/TSX-1.SAR.L1B/TDX1_SAR__SSC______SM_S_SRA_20180724T101928_20180724T101938/TDX1_SAR__SSC______SM_S_SRA_20180724T101928_20180724T101938.xml'
    filename_slave = '/Users/timo/Documents/WuhanEast/dims_op_oc_dfd2_675921665_2/TSX-1.SAR.L1B/TDX1_SAR__SSC______SM_S_SRA_20180826T101930_20180826T101940/TDX1_SAR__SSC______SM_S_SRA_20180826T101930_20180826T101940.xml'
    filepath_out = '/Users/timo/Documents/WuhanEast/pysar/'
    slc_master = slc.fromTSX(filename_master, 0)
    slc_slave = slc.fromTSX(filename_slave, 0)

    pair = insar_pair.createInSarPair(slc_master, slc_slave)

    print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')

    pair.save(insar_pair.createFilename(pair, filepath_out), savetiff=True)