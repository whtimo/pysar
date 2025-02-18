from scipy.interpolate import NearestNDInterpolator

from pysar import footprint, coordinates
from pysar.insar import resampled_pair, baseline
from pysar.sar import metadata, cpl_float_memory_slcdata
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.constants import c
from rasterio.windows import Window
import rasterio
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib


class TopoInterferogram:
    def __init__(self, filepath: str = None):
        self.master_metadata = None
        self.slave_metadata = None
        self.perpendicular_baseline = None
        self.temporal_baseline = None
        self.interferogram_tiff_path = None
        self.data = None
        self._interfero_data = None

        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("Interferogram")
            if pair_elem:
                self.perpendicular_baseline = float(pair_elem.attrib['perpendicular_baseline'])
                self.temporal_baseline = int(pair_elem.attrib['temporal_baseline'])
                self.master_metadata = metadata.fromXml(pair_elem.find("Master/MetaData"))
                self.slave_metadata = metadata.fromXml(pair_elem.find("Slave/MetaData"))
                self.interferogram_tiff_path = pathlib.Path(filepath).parent / pair_elem.find("FilePath").text

    def save(self, directory: str = None, filename: str = None, tiff_filename: str = None, overwrite: bool = True,
             filtered: bool = False):

        xml_filename = ''
        insar_tiff_fn = ''

        filter = ''
        if filtered:
            filter = 'filtered.'

        if directory is None:
            if filename is not None and tiff_filename is not None:
                xml_filename = pathlib.Path(filename)
                insar_tiff_fn = pathlib.Path(tiff_filename)
        else:
            counter = 0
            xml_filename = pathlib.Path(
                directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.pysar.{filter}topo.interfero.xml'
            insar_tiff_fn = pathlib.Path(
                directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.{filter}topo.interfero.tiff'

            while not overwrite and xml_filename.exists():
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.pysar.{filter}topo.interfero.xml'
                insar_tiff_fn = pathlib.Path(
                    directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.{filter}topo.interfero.tiff'

        if len(str(xml_filename)) > 0:
            root = ET.Element("PySar")
            pair_elem = ET.SubElement(root, "Interferogram")
            pair_elem.attrib["perpendicular_baseline"] = str(self.perpendicular_baseline)
            pair_elem.attrib["temporal_baseline"] = str(self.temporal_baseline)
            master_elem = ET.SubElement(pair_elem, "Master")
            self.master_metadata.toXml(master_elem)
            slave_elem = ET.SubElement(pair_elem, "Slave")
            self.slave_metadata.toXml(slave_elem)

            if overwrite or not insar_tiff_fn.exists():
                metadata = {
                    "driver": "GTiff",  # GeoTIFF format
                    "height": self.getHeight(),  # Number of rows
                    "width": self.getWidth(),  # Number of columns
                    "count": 1,  # Single band (complex data)
                    "dtype": np.complex64,  # Complex float32 data type
                    "transform": rasterio.Affine.identity(),  # Identity transform (no georeferencing)
                }

                with rasterio.open(insar_tiff_fn, "w", **metadata) as dst:
                    dst.write(self.read(), 1)  # Write the complex array to the first band

            file_path_elem = ET.SubElement(pair_elem, "FilePath")
            file_path_elem.text = str(insar_tiff_fn.relative_to(xml_filename.parent))
            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            # Write the pretty-printed XML to a file
            with open(xml_filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

    def __openfile(self):
        if self.data is None and self.interferogram_tiff_path is not None:
            self.data = rasterio.open(self.interferogram_tiff_path)

    def getWidth(self) -> int:
        self.__openfile()
        if self.data is None:
            return self._interfero_data.shape[1]
        else:
            return self.data.width

    def getHeight(self) -> int:
        self.__openfile()
        if self.data is None:
            return self._interfero_data.shape[0]
        else:
            return self.data.height

    def read(self, window: Window = None) -> np.ndarray:
        self.__openfile()
        if self.data is None:
            if window is None:
                return self._interfero_data
            else:
                return self._interfero_data[window.row_off:window.row_off+window.height,window.col_off:window.col_off+window.width]
        else:
            if window is None:
                return self.data.read(1)
            else:
                return self.data.read(1, window=window)

    def subset(self, window: Window):
        data = self.read(window)
        mem = cpl_float_memory_slcdata.CplFloatMemorySlcData(self.data)
        return mem

    def multilook(self, multilook_range=1, multilook_azimuth=1):
        data = self.read()

        if data.shape[0] % multilook_azimuth != 0:
            data = data[0:-1, :]

        if data.shape[1] % multilook_range != 0:
            data = data[:, 0:-1]

        # Reshape the array into blocks of size y (rows) and x (columns)
        reshaped = data.reshape(data.shape[0] // multilook_azimuth, multilook_azimuth, data.shape[1] // multilook_range,
                                multilook_range)
        # Average over the y and x dimensions
        reduced = reshaped.mean(axis=(1, 3))
        mem = cpl_float_memory_slcdata.CplFloatMemorySlcData(reduced)
        return mem


def createTopoInterferogram(master: metadata.MetaData, slave: metadata.MetaData, interfero_data: np.ndarray,
                            base_line: baseline.Baseline = None):
    interfero = TopoInterferogram()
    interfero.master_metadata = master
    interfero.slave_metadata = slave
    if base_line is None:
        base_line = baseline.Baseline(master, slave)

    interfero.perpendicular_baseline = base_line.perpendicular_baseline(master.number_columns / 2,
                                                                        master.number_rows / 2)
    interfero.temporal_baseline = base_line.temporal_baseline

    interfero._interfero_data = interfero_data

    return interfero

def fromBzarXml(xml_path: str) -> TopoInterferogram:
    interfero = TopoInterferogram()

    root = ET.parse(xml_path).getroot()
    pair_elem = root.find("Interferogram")
    if pair_elem:
        interfero.perpendicular_baseline = float(pair_elem.attrib['baseline'])
        interfero.temporal_baseline = int(pair_elem.attrib['temp_baseline'])
        interfero.master_metadata = metadata.fromBzarXml(pair_elem.find("MasterSlcImage/Band"))
        interfero.slave_metadata = metadata.fromBzarXml(pair_elem.find("SlaveSlcImage/Band"))
        interfero.interferogram_tiff_path = pathlib.Path(xml_path).parent / pair_elem.find("FilePath").text
    return interfero

def interpolate_phase_residuals_natural_neighbor(samples, lines, values, grid_size):
    """
    Interpolate phase residuals using Natural Neighbor interpolation

    Parameters:
    -----------
    samples : array-like
        Sample coordinates of the PS points
    lines : array-like
        Line coordinates of the PS points
    values : array-like
        Unwrapped phase residual values
    grid_size : tuple
        Size of the output grid (n_lines, n_samples)

    Returns:
    --------
    numpy.ndarray : Interpolated grid
    """

    # Create interpolator
    interpolator = NearestNDInterpolator(list(zip(samples, lines)), values)

    # Create regular grid
    grid_lines, grid_samples = np.mgrid[0:grid_size[0], 0:grid_size[1]]

    # Interpolate
    interpolated = interpolator(grid_samples, grid_lines)

    return interpolated

def remove_topographic_phases(flat_interfero: np.ndarray, topo_phases: np.ndarray) -> np.ndarray:
    result = flat_interfero * np.conj(topo_phases)
    return result

def get_topographic_phases(master_metadata: metadata.MetaData, slave_metadata:metadata.MetaData, dem_path: str) -> np.ndarray:

    with rasterio.open(dem_path) as srtm:
        # Convert the transformed coordinates to row/column indices in the raster
        srtm_transform = srtm.transform
        srtm_data = np.array(srtm.read(1), dtype=np.float32)

    tl_x, tl_y = ~srtm_transform * (master_metadata.footprint.left(), master_metadata.footprint.top())
    br_x, br_y = ~srtm_transform * (master_metadata.footprint.right(), master_metadata.footprint.bottom())
    min_x = int(np.min([tl_x, br_x])) - 1
    min_y = int(np.min([tl_y, br_y])) - 1
    max_x = int(np.max([tl_x, br_x])) + 1
    max_y = int(np.max([tl_y, br_y])) + 1
    if min_x < 0: min_x = 0
    if min_y < 0: min_y = 0
    if max_x >= srtm_data.shape[1]: max_x = srtm_data.shape[1] - 1
    if max_y >= srtm_data.shape[0]: max_y = srtm_data.shape[0] - 1
    range_x = max_x - min_x
    range_y = max_y - min_y

    x = np.linspace(min_x, max_x, range_x, dtype=np.int32)
    y = np.linspace(min_y, max_y, range_y, dtype=np.int32)
    x_grid, y_grid = np.meshgrid(x, y)
    lons, lats = srtm_transform * (x_grid, y_grid)
    heights = map_coordinates(srtm_data, [[y_grid], [x_grid]], order=1, mode='nearest')[0]

    print('Transform into geocentric coordinates')
    geoc = coordinates.geodetic_to_geocentric(lats, lons, heights)

    print('Calculate SAR image coordinates')
    sar_mx, sar_my = master_metadata.pixel_from_geocentric(geoc)
    sar_sx, sar_sy = slave_metadata.pixel_from_geocentric(geoc)

    master_range_time = master_metadata._burst.pixel_to_range_time(sar_mx)
    slave_range_time = slave_metadata._burst.pixel_to_range_time(sar_sx)

    master_range = master_range_time * c
    slave_range = slave_range_time * c
    delta_r = master_range - slave_range
    topo_phase = (4 * np.pi * delta_r) / master_metadata.wavelength

    grid_size = (master_metadata.number_rows, master_metadata.number_columns)
    interpolate = interpolate_phase_residuals_natural_neighbor(sar_mx.reshape(-1), sar_my.reshape(-1), topo_phase.reshape(-1), grid_size)

    interpolate_cpl = 1 + 1j * interpolate

    return interpolate_cpl