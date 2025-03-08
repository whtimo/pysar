from pysar.sar import slc
import numpy as np
from rasterio.windows import Window
import matplotlib.pyplot as plt

if __name__ == "__main__":

    filename = ''
    slc = slc.fromTSX(filename, 0)
    newslc = slc.subset(window=Window(1000, 2000, 1024, 1024*4))
    newslcml = newslc.multilook(1, 4)
    data = newslcml.slcdata.read()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(np.abs(data),cmap='gray', vmin=0, vmax=256)
    plt.show()

