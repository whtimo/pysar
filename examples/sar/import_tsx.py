from pysar.sar import slc

if __name__ == "__main__":

    filename = ''
    out_filename = ''

    slc = slc.fromTSX(filename, 0)
    slc.save(directory=out_filename)

