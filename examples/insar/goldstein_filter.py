from pysar import tools
from pysar.insar import flat_interferogram
from pysar.sar import filter

if __name__ == "__main__":

    file_name = ''
    output_path = ''

    flat = flat_interferogram.FlatInterferogram(file_name)
    filtered = filter.goldstein_filter(flat.read())

    filty = flat_interferogram.createFlatInterferogram(flat.master_metadata, flat.slave_metadata, filtered)
    filty.save(directory=output_path, filtered=True, output=tools.output_console)
