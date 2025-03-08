import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pysar.insar import insar_pair

if __name__ == "__main__":

    file_pair = ''
    resample_tiff_file = ''

    pair = insar_pair.InSarPair(file_pair)
    resample_tiff = rasterio.open(resample_tiff_file)
    resample_data = resample_tiff.read(1)
    resample_sub = resample_data[0:1000, 0:1000]

    master_data = pair.master.slcdata.read()
    master_sub = master_data[0:1000, 0:1000]

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