from pysar import flat_interferogram, filter, tools


if __name__ == "__main__":

    file_name = '/Users/timo/Documents/WuhanEast/pysar/TDX-1_0_2018-07-24__TDX-1_2018-08-26.pysar.flat.interfero.xml'
    output_path = '/Users/timo/Documents/WuhanEast/pysar'

    flat = flat_interferogram.FlatInterferogram(file_name)
    filtered = filter.goldstein_filter(flat.read())

    filty = flat_interferogram.createFlatInterferogram(flat.master_metadata, flat.slave_metadata, filtered)
    filty.save(directory=output_path, filtered=True, output=tools.output_console)
