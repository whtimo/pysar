import rasterio
from pysar import islcdata

#Single Swath and Single Burst Slc Data
class CplFloatSlcData(islcdata.ISlcData):
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
