import xml.etree.ElementTree as ET
import pathlib
from pysar import metadata, cpl_float_slcdata


# Single Swath - Single Burst Slc
class Slc:
    def __init__(self):
        self.metadata = None
        self.slcdata = None

def fromTSX(xml_path: str, swath_id : int) -> Slc:
    slc = Slc()
    pol_file_list = getPolCosFileNamesFromTsx(xml_path)
    slc.metadata = metadata.fromTSX(xml_path, pol_file_list[swath_id][1])
    slc.slcdata = cpl_float_slcdata.CplFloatSlcData(pol_file_list[swath_id][0])
    return slc

def numberOfSwathsFromTsx(xml_path: str) -> int:
    list = getPolCosFileNamesFromTsx(xml_path)
    return len(list)


def getPolCosFileNamesFromTsx(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = []
    xml = pathlib.Path(xml_path)

    productComponents = root.find('productComponents')
    for image_data_element in productComponents.findall('imageData'):
        pol_layer = image_data_element.find('polLayer').text  # Extract <polLayer>

        # Extract <path> and <filename> and combine them
        location = image_data_element.find('file/location')
        path = pathlib.Path(location.find('path').text)
        filename = pathlib.Path(location.find('filename').text)
        full_path = xml.parent/path/filename  # Combine path and filename
        result.append((full_path, pol_layer))

    return result