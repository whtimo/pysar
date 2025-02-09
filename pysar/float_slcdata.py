import numpy as np
import rasterio
from pysar import islcdata, float_memory_slcdata
import xml.etree.ElementTree as ET
import pathlib
from rasterio.windows import Window


#Single Swath and Single Burst Slc Data
class FloatSlcData(islcdata.ISlcData):
    def __init__(self, filename : str):
        self.filename = filename
        self.data = None

    def __openfile(self):
        if self.data is None:
            self.data = rasterio.open(self.filename)

    def getWidth(self) -> int:
        self.__openfile()
        return self.data.width

    def getHeight(self) -> int:
        self.__openfile()
        return self.data.height


    def read(self, window: Window = None) -> np.ndarray:
        self.__openfile()
        if window is None:
            return self.data.read(1)
        else:
            return self.data.read(1, window=window)

    def subset(self, window: Window):
        data = self.read(window)
        mem = float_memory_slcdata.FloatMemorySlcData(data)
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
        mem = float_memory_slcdata.FloatMemorySlcData(reduced)
        return mem

def fromXml(root: ET.Element, xml_file_path:str) -> FloatSlcData:
    file_path_elem = root.find("FilePath")
    file_name = file_path_elem.text
    fullpath = pathlib.Path(xml_file_path).parent / file_name
    slc = FloatSlcData(fullpath)
    return slc
