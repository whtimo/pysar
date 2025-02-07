import sys
sys.path.extend(['/Users/timo/src/pysar'])

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pysar import insar_pair

if __name__ == "__main__":

    file_pair = '/Users/timo/Documents/WuhanEast/pysar/TDX-1_2018-07-24__2018-08-26.pysar.pair.xml'

    resample_tiff_file = '/Users/timo/Documents/WuhanEast/pysar/resample.tiff'

    pair = insar_pair.InSarPair(file_pair)
    resample_tiff = rasterio.open(resample_tiff_file)
    resample_data = resample_tiff.read(1)
    resample_sub = resample_data[15000:16024, 10000:11024]

    master_data = pair.master.slcdata.read()
    master_sub = master_data[15000:16024, 10000:11024]

    inter = master_sub * np.conjugate(resample_sub)
    # Plot the image with geographic coordinates
    plt.figure(figsize=(8, 6))
    img = plt.imshow(abs(resample_sub), cmap='gray', vmin=0, vmax=128)
    plt.show()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(abs(master_sub), cmap='gray', vmin=0, vmax=128)
    plt.show()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(np.angle(inter), cmap='hsv')
    plt.show()