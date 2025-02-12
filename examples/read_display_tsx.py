#import sys
#sys.path.extend(['/Users/timo/src/pysar'])

import matplotlib.pyplot as plt
from pysar.sar import slc
import numpy as np

if __name__ == "__main__":

    filename = '/Users/timo/Documents/Rapa Nui/dims_op_oc_dfd2_693810856_1/TSX-1.SAR.L1B/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450.xml'
    slc = slc.fromTSX(filename, 0)
    data = slc.slcdata.read()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(np.abs(data),cmap='gray', vmin=0, vmax=256)
    plt.show()
