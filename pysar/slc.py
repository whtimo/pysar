import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib
import numpy as np
from pysar import metadata, cpl_float_slcdata
from rasterio.windows import Window, bounds

# Single Swath - Single Burst Slc
class Slc:
    def __init__(self):
        self.metadata = None
        self.slcdata = None

    def subset(self, window: Window):
        newslc = Slc()
        newslc.metadata = self.metadata.subset(window)
        newslc.slcdata = self.slcdata.subset(window)
        return newslc

    def multilook(self, multilook_range = 1, multilook_azimuth = 1):
        newslc = Slc()
        newslc.metadata = self.metadata.multilook(multilook_range, multilook_azimuth)
        newslc.slcdata = self.slcdata.multilook(multilook_range, multilook_azimuth)
        return newslc

def fromTSX(xml_path: str, swath_id: int) -> Slc:
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
        full_path = xml.parent / path / filename  # Combine path and filename
        result.append((full_path, pol_layer))

    return result


def saveToPysarXml(slc, xml_path: str, slcdata: np.ndarray):
    root = ET.Element("PySar")
    slc_elem = ET.SubElement(root, "Slc")
    slc.metadata.toXml(slc_elem)
    tiff_name = slc.metadata.acquisition_date.isoformat() + ".slc.tiff"
    slc.slcdata.toXml(slc_elem, tiff_name, xml_path, slcdata)
    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

    # Write the pretty-printed XML to a file
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

def fromPysarXml(xml_path: str) -> Slc:
    slc = Slc()

    root = ET.parse(xml_path).getroot()
    slc_elem = root.find("Slc")
    if slc_elem is None: return None

    slc.metadata = metadata.fromXml(slc_elem.find("MetaData"))
    slc.slcdata = cpl_float_slcdata.fromXml(slc_elem, xml_path)
    return slc

def getPysarPathName(slc: Slc, directory: str, overwrite=False) -> pathlib.Path:
    path = pathlib.Path(directory)
    if path.exists():
        counter = 0
        fullpath = path / f'{slc.metadata.sensor}_{counter}_{slc.metadata.acquisition_date.isoformat()}.pyear.slc.xml'
        while not overwrite and fullpath.exists():
            counter += 1
            fullpath = path / f'{slc.metadata.sensor}_{counter}_{slc.metadata.acquisition_date.isoformat()}.pyear.slc.xml'

        return fullpath

def fromBzarXml(xml_path: str) -> Slc:
    slc = Slc()

    root = ET.parse(xml_path).getroot()
    slc_elem = root.find("SlcImage")
    if slc_elem is None: return None

    slc.metadata = metadata.fromBzarXml(slc_elem.find("Band"))
    slc.slcdata = cpl_float_slcdata.fromXml(slc_elem, xml_path)
    return slc