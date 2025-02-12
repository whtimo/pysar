from pysar.sar import slc

if __name__ == "__main__":

    filename = '/Users/timo/Documents/Rapa Nui/dims_op_oc_dfd2_693810856_1/TSX-1.SAR.L1B/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450.xml'
    out_filename = '/Users/timo/Documents/WuhanEast/pysar'
    slc = slc.fromTSX(filename, 0)
    slc.save(directory=out_filename)

