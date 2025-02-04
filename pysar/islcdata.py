import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import rasterio

from rasterio.windows import Window

#Single Swath and Single Burst Slc Data
class ISlcData:
    def getWidth(self) -> int:
        return 0

    def getHeight(self) -> int:
        return 0



    def toXml(self, root: ET.Element, relative_path_to_filename: str, xml_file_path:str = None, complexData:np.ndarray = None):
        if complexData is not None and xml_file_path is not None:
            absolutePath = pathlib.Path(xml_file_path).parent / relative_path_to_filename
            metadata = {
                "driver": "GTiff",  # GeoTIFF format
                "height": complexData.shape[0],  # Number of rows
                "width": complexData.shape[1],  # Number of columns
                "count": 1,  # Single band (complex data)
                "dtype": np.complex64,  # Complex float32 data type
                "transform": rasterio.Affine.identity(),  # Identity transform (no georeferencing)
            }

            with rasterio.open(absolutePath, "w", **metadata) as dst:
                dst.write(complexData, 1)  # Write the complex array to the first band

        slc_elem = ET.SubElement(root, "slcdata")
        file_path_elem = ET.SubElement(root, "path")
        file_path_elem.text = relative_path_to_filename