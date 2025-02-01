import rasterio
from pysar import islcdata

#Single Swath and Single Burst Slc Data
class CplFloatSlcData(islcdata.ISlcData):
    def __init__(self, filename : str):
        self.filename = filename
        self.data = rasterio.open(filename)

    def getWidth(self) -> int:
        return self.data.width

    def getHeight(self) -> int:
        return self.data.height
