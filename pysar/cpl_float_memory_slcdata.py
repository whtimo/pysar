import numpy as np
from pysar import islcdata
from rasterio.windows import Window

#Single Swath and Single Burst Slc Data
class CplFloatMemorySlcData(islcdata.ISlcData):
    def __init__(self, data: np.ndarray):
        self.data = data


    def getWidth(self) -> int:
        return self.data.shape[1]

    def getHeight(self) -> int:
        return self.data.shape[0]

    def read(self, window: Window = None) -> np.ndarray:
        if window is None:
            return self.data
        else:
            return self.data[window.col_off:window.col_off+window.width, window.row_off:window.row_off+window.height]

    def subset(self, window: Window):
        data = self.read(window)
        mem = cpl_float_memory_slcdata.CplFloatMemorySlcData(data)
        return mem

    def multilook(self, multilook_range=1, multilook_azimuth=1):
        data = self.read()

        if data.shape[0] % multilook_azimuth != 0:
            data = data[0:-1,:]

        if data.shape[1] % multilook_range != 0:
            data = data[:,0:-1]

        # Reshape the array into blocks of size y (rows) and x (columns)
        reshaped = data.reshape(data.shape[0] // multilook_azimuth, multilook_azimuth, data.shape[1] // multilook_range, multilook_range)
        # Average over the y and x dimensions
        reduced = reshaped.mean(axis=(1, 3))
        mem = CplFloatMemorySlcData(reduced)
        return mem