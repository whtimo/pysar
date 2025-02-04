import sys
sys.path.extend(['/Users/timo/src/pysar'])

import matplotlib.pyplot as plt
from pysar import slc
import numpy as np

if __name__ == "__main__":

    filename = '/Users/timo/Documents/tsx_rapanui.pysar.xml'
    slc = slc.fromPysarXml(filename)
    data = slc.slcdata.read()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(np.abs(data),cmap='gray', vmin=0, vmax=256)
    plt.show()
