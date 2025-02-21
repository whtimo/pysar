#import sys
#sys.path.extend(['/Users/timo/src/pysar'])

import matplotlib.pyplot as plt
from pysar.sar import slc
import numpy as np

if __name__ == "__main__":

    filename = '/home/timo/Data/SouthAfrica_2/cossc/dims_op_oc_dfd2_696749819_1/TDM.SAR.COSSC/1967680_002/TDM1_SAR__COS_BIST_SM_S_SRA_20230303T162519_20230303T162527/TDX1_SAR__SSC_BTX1_SM_S_SRA_20230303T162519_20230303T162527/TDX1_SAR__SSC_BTX1_SM_S_SRA_20230303T162519_20230303T162527.xml'
    slc = slc.fromTSX(filename, 0)
    data = slc.slcdata.read()

    plt.figure(figsize=(8, 6))
    img = plt.imshow(np.abs(data),cmap='gray', vmin=0, vmax=256)
    plt.show()
