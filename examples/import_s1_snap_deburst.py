from pysar.sar import slc, metadata


if __name__ == "__main__":

    file_name = '/home/timo/Data/LVS1_pysar/single/S1A_IW_SLC__1SDV_20230127T134400_20230127T134428_046970_05A22D_7B8E_split_deb.dim'
    output_path = '/home/timo/Data/LVS1_pysar'

    slc = slc.fromDim(file_name)
    slc.save(directory=output_path)