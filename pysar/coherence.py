from pysar import metadata, footprint, resampled_pair, coordinates, cpl_float_memory_slcdata,baseline
import numpy as np
from rasterio.windows import Window
import rasterio
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib
from scipy.ndimage import uniform_filter

class Coherence:
    def __init__(self, filepath:str = None):
        self.master_metadata = None
        self.slave_metadata = None
        self.coherence_tiff_path = None
        self.data = None
        self._coherence_data = None

        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("Coherence")
            if pair_elem:
                self.master_metadata = metadata.fromXml(pair_elem.find("Master/MetaData"))
                self.slave_metadata = metadata.fromXml(pair_elem.find("Slave/MetaData"))

                self.coherence_tiff_path = pathlib.Path(filepath).parent / pair_elem.find("FilePath").text

    def save(self, directory: str = None, filename: str = None, tiff_filename: str = None, overwrite: bool = True,):

        xml_filename = ''
        coherence_tiff_fn = ''

        if directory is None:
            if filename is not None and tiff_filename is not None:
                xml_filename = pathlib.Path(filename)
                coherence_tiff_fn = pathlib.Path(tiff_filename)
        else:
            counter = 0
            xml_filename = pathlib.Path(
                directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.pysar.coherence.xml'
            coherence_tiff_fn = pathlib.Path(
                directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.coherence.tiff'

            while not overwrite and xml_filename.exists():
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.pysar.coherence.xml'
                coherence_tiff_fn = pathlib.Path(
                    directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.coherence.tiff'

        if len(str(xml_filename)) > 0:

            root = ET.Element("PySar")
            pair_elem = ET.SubElement(root, "Coherence")
            master_elem = ET.SubElement(pair_elem, "Master")
            self.master_metadata.toXml(master_elem)
            slave_elem = ET.SubElement(pair_elem, "Slave")
            self.slave_metadata.toXml(slave_elem)

            if overwrite or not coherence_tiff_fn.exists():
                metadata = {
                    "driver": "GTiff",  # GeoTIFF format
                    "height": self.getHeight(),  # Number of rows
                    "width": self.getWidth(),  # Number of columns
                    "count": 1,  # Single band (complex data)
                    "dtype": np.float32,  # Complex float32 data type
                    "transform": rasterio.Affine.identity(),  # Identity transform (no georeferencing)
                }

                with rasterio.open(coherence_tiff_fn, "w", **metadata) as dst:
                    dst.write(self.read(), 1)  # Write the complex array to the first band

            file_path_elem = ET.SubElement(pair_elem, "FilePath")
            file_path_elem.text = str(coherence_tiff_fn.relative_to(xml_filename.parent))
            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            # Write the pretty-printed XML to a file
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml)


    def __openfile(self):
        if self.data is None and self.coherence_tiff_path is not None:
            self.data = rasterio.open(self.coherence_tiff_path)

    def getWidth(self) -> int:
        self.__openfile()
        if self.data is None:
            return self._coherence_data.shape[1]
        else:
            return self.data.width

    def getHeight(self) -> int:
        self.__openfile()
        if self.data is None:
            return self._coherence_data.shape[0]
        else:
            return self.data.height

    def read(self, window: Window = None) -> np.ndarray:
        self.__openfile()
        if self.data is None:
            if window is None:
                return self._coherence_data
            else:
                return self._coherence_data[window.row_off:window.row_off+window.height,window.col_off:window.col_off+window.width]
        else:
            if window is None:
                return self.data.read(1)
            else:
                return self.data.read(1, window=window)

def createCoherence(master: metadata.MetaData, slave: metadata.MetaData, coherence_data: np.ndarray):
    coh = Coherence()
    coh.master_metadata = master
    coh.slave_metadata = slave
    coh._coherence_data = coherence_data

    return coh

def compute_coherence(master, slave, flat_interfero: np.ndarray = None, window_size=5):
    """
    Compute the coherence image from two coregistered complex SAR images, processing line by line.

    Parameters:
        master (np.ndarray): Complex master SAR image.
        slave (np.ndarray): Complex slave SAR image.
        flat_interfero:(np.ndarray): Flat interferogram for better coherence (optional)
        window_size (int): Size of the moving window for spatial averaging.

    Returns:
        coherence (np.ndarray): Coherence image, with values in the range [0, 1].
    """
    # Ensure the images have the same shape
    if master.shape != slave.shape:
        raise ValueError("Master and slave images must have the same shape.")

    # Get the shape of the input images
    rows, cols = master.shape

    # Initialize the output coherence image
    coherence = np.zeros((rows, cols), dtype=np.float32)

    # Pad the images to handle edges when applying the moving window
    pad_size = window_size // 2

    # Process each line
    for y in range(rows):
        print(f'Coherence processing: {y} / {rows}')
        if y-pad_size >= 0 and y-pad_size < rows:
            # Extract the current line and its neighborhood (window_size x window_size)
            master_line = master[y-pad_size:y + pad_size+1, :]
            slave_line = slave[y - pad_size:y + pad_size+1, :]
            if flat_interfero is None:
                # Compute the interferogram for the current line
                interferogram_line = master_line * np.conj(slave_line)
            else:
                interferogram_line = flat_interfero[y - pad_size:y + pad_size+1, :]

            # Compute the intensity of the master and slave for the current line
            intensity_master_line = np.abs(master_line) ** 2
            intensity_slave_line = np.abs(slave_line) ** 2

            # Apply a uniform filter (moving window) to compute spatial averages
            avg_interferogram_line = uniform_filter(interferogram_line, size=(1, window_size), mode='constant')
            avg_intensity_master_line = uniform_filter(intensity_master_line, size=(1, window_size), mode='constant')
            avg_intensity_slave_line = uniform_filter(intensity_slave_line, size=(1, window_size), mode='constant')

            coherence_line = np.abs(avg_interferogram_line[pad_size, :]) / np.sqrt(
                     avg_intensity_master_line[pad_size, :] * avg_intensity_slave_line[pad_size, :]
                 )

            # Store the coherence for the current line
            coherence[y, :] = coherence_line

    return coherence

