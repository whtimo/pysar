import numpy as np
import rasterio
from pysar.sar import islcdata, cpl_float_memory_slcdata
import xml.etree.ElementTree as ET
import pathlib
from rasterio.windows import Window


#Single Swath and Single Burst Slc Data
class IqFloatSlcData(islcdata.ISlcData):
    def __init__(self, filename_i : str, filename_q : str):
        self.filename_i = filename_i
        self.filename_q = filename_q
        self.data_i = None
        self.data_q = None

    def __openfile(self):
        if self.data_i is None or self.data_q is None:
            self.data_i = rasterio.open(self.filename_i)
            self.data_q = rasterio.open(self.filename_q)

    def getWidth(self) -> int:
        self.__openfile()
        return self.data_i.width

    def getHeight(self) -> int:
        self.__openfile()
        return self.data_i.height


    def read(self, window: Window = None) -> np.ndarray:
        self.__openfile()
        if window is None:
            i = self.data_i.read(1)
            q = self.data_q.read(1)
            return i + 1j * q
        else:
            i = self.data_i.read(1, window=window)
            q = self.data_q.read(1, window=window)
            return i + 1j * q

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
        mem = cpl_float_memory_slcdata.CplFloatMemorySlcData(reduced)
        return mem


