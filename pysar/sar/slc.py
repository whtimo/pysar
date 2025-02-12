import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib
from pysar.sar import cpl_float_slcdata, metadata
from rasterio.windows import Window


class Slc:
    """
    Single-look complex SAR data. This class supports only one swath and one burst
    """

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

    def save(self,  directory: str = None, filename: str = None, tiff_filename: str = None, overwrite: bool = False):
        """
        Saves the SLC data

        :param directory: Save into this directory with automatic created filenames (defaut)
        :param filename: Filename for the xml file (requires the directory to not be set and the tiff_filename)
        :param tiff_filename: Filename for the tiff file (requires the directory to not be set and the filename)
        """

        xml_filename = ''
        tiff_fn = ''

        if directory is None:
            if not tiff_filename is None and not filename is None:
                xml_filename = filename
                tiff_fn = tiff_filename
        else:
            counter = 0
            xml_filename = pathlib.Path(directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.pysar.slc.xml'
            tiff_fn = pathlib.Path(directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.slc.tiff'
            while not overwrite and (xml_filename.exists() or tiff_fn.exists()):
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.pysar.slc.xml'
                tiff_fn = pathlib.Path(
                    directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.slc.tiff'

        if not xml_filename is None:
            root = ET.Element("PySar")
            slc_elem = ET.SubElement(root, "Slc")
            self.metadata.toXml(slc_elem)
            if overwrite or not pathlib.Path(tiff_fn).exists():
                self.slcdata.saveTiff(tiff_fn, self.slcdata.read())
            self.slcdata.toXml(slc_elem, pathlib.Path(tiff_fn).relative_to(pathlib.Path(xml_filename).parent))
            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            # Write the pretty-printed XML to a file
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml)


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


def fromPysarXml(xml_path: str) -> Slc:
    slc = Slc()

    root = ET.parse(xml_path).getroot()
    slc_elem = root.find("Slc")
    if slc_elem is None: return None

    slc.metadata = metadata.fromXml(slc_elem.find("MetaData"))
    slc.slcdata = cpl_float_slcdata.fromXml(slc_elem, xml_path)
    return slc

def fromBzarXml(xml_path: str) -> Slc:
    slc = Slc()

    root = ET.parse(xml_path).getroot()
    slc_elem = root.find("SlcImage")
    if slc_elem is None: return None

    slc.metadata = metadata.fromBzarXml(slc_elem.find("Band"))
    slc.slcdata = cpl_float_slcdata.fromXml(slc_elem, xml_path)
    return slc