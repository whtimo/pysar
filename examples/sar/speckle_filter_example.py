from pysar.sar import slc, filter
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    file_name = ''

    slc = slc.fromTSX(file_name, 0)

    data = np.abs(slc.slcdata.read(window=Window(12800, 30000, 3000, 3000)))
    filtered = filter.non_local_means_filter(data[1200:1500,2080:2480], patch_size=7, search_window_size=21, h=100)

    #fig, (ax1) = plt.subplots(1, 1, figsize=(10, 12))
    #img = ax1.imshow(data, cmap='gray', vmin=0, vmax=256)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    img = ax1.imshow(data[1200:1500,2080:2480], cmap='gray', vmin=0, vmax=256)
    img2 = ax2.imshow(filtered, cmap='gray', vmin=0, vmax=256)
    plt.tight_layout()
    plt.show()
