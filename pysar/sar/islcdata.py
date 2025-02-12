import numpy as np
import xml.etree.ElementTree as ET
import rasterio

#Single Swath and Single Burst Slc Data
class ISlcData:
    def getWidth(self) -> int:
        return 0

    def getHeight(self) -> int:
        return 0

    def saveTiff(self, filename: str, complexData:np.ndarray):
        metadata = {
            "driver": "GTiff",  # GeoTIFF format
            "height": complexData.shape[0],  # Number of rows
            "width": complexData.shape[1],  # Number of columns
            "count": 1,  # Single band (complex data)
            "dtype": np.complex64,  # Complex float32 data type
            "transform": rasterio.Affine.identity(),  # Identity transform (no georeferencing)
        }

        with rasterio.open(filename, "w", **metadata) as dst:
            dst.write(complexData, 1)  # Write the complex array to the first band

    def saveFloatTiff(self, filename: str, floatData:np.ndarray):
        metadata = {
            "driver": "GTiff",  # GeoTIFF format
            "height": floatData.shape[0],  # Number of rows
            "width": floatData.shape[1],  # Number of columns
            "count": 1,  # Single band (complex data)
            "dtype": np.float32,  # Complex float32 data type
            "transform": rasterio.Affine.identity(),  # Identity transform (no georeferencing)
        }

        with rasterio.open(filename, "w", **metadata) as dst:
            dst.write(floatData, 1)  # Write the complex array to the first band


    def toXml(self, root: ET.Element, relative_path_to_filename: str):
        file_path_elem = ET.SubElement(root, "FilePath")
        file_path_elem.text = str(relative_path_to_filename)