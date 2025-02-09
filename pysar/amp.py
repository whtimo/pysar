import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib
import numpy as np
from pysar import metadata, cpl_float_slcdata
from rasterio.windows import Window, bounds

class Amp:
    """
    Single-look amplitude SAR data. This class supports only one swath and one burst
    """

    def __init__(self):
        self.metadata = None
        self.ampdata = None

    def subset(self, window: Window):
        newslc = Amp()
        newslc.metadata = self.metadata.subset(window)
        newslc.ampdata = self.ampdata.subset(window)
        return newslc

    def multilook(self, multilook_range = 1, multilook_azimuth = 1):
        newslc = Amp()
        newslc.metadata = self.metadata.multilook(multilook_range, multilook_azimuth)
        newslc.ampdata = self.ampdata.multilook(multilook_range, multilook_azimuth)
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
            xml_filename = pathlib.Path(directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.pysar.amp.xml'
            tiff_fn = pathlib.Path(directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.amp.tiff'
            while not overwrite and (xml_filename.exists() or tiff_fn.exists()):
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.pysar.amp.xml'
                tiff_fn = pathlib.Path(
                    directory) / f'{self.metadata.sensor}_{counter}_{self.metadata.acquisition_date.isoformat()}.amp.tiff'

        if not xml_filename is None:
            root = ET.Element("PySar")
            slc_elem = ET.SubElement(root, "Amplitude")
            self.metadata.toXml(slc_elem)
            if overwrite or not pathlib.Path(tiff_fn).exists():
                self.ampdata.saveFloatTiff(tiff_fn, self.ampdata.read())
            self.ampdata.toXml(slc_elem, pathlib.Path(tiff_fn).relative_to(pathlib.Path(xml_filename).parent))
            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            # Write the pretty-printed XML to a file
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

def createAmp(meta: metadata.MetaData, amp_data: np.ndarray):
    amp = Amp()
    amp.metadata = meta
    amp.ampdata = np.abs(amp_data)

    return amp



