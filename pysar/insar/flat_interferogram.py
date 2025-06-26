from pysar import footprint, coordinates
from pysar.insar import resampled_pair, baseline
from pysar.sar import metadata, cpl_float_memory_slcdata
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from rasterio.windows import Window
import rasterio
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib

#import matplotlib.pyplot as plt

class FlatInterferogram:
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
                directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{counter}_{self.slave_metadata.acquisition_date.isoformat()}.pysar.{filter}flat.interfero.xml'
            insar_tiff_fn = pathlib.Path(
                directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{counter}_{self.slave_metadata.acquisition_date.isoformat()}.{filter}flat.interfero.tiff'

            while not overwrite and xml_filename.exists():
                counter += 1
                xml_filename = pathlib.Path(
                    directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{counter}_{self.slave_metadata.acquisition_date.isoformat()}.pysar.{filter}flat.interfero.xml'
                insar_tiff_fn = pathlib.Path(
                    directory) / f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{counter}_{self.slave_metadata.acquisition_date.isoformat()}.{filter}flat.interfero.tiff'

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


def createFlatInterferogram(master: metadata.MetaData, slave: metadata.MetaData, flat_data: np.ndarray,
                            base_line: baseline.Baseline = None):
    interfero = FlatInterferogram()
    interfero.master_metadata = master
    interfero.slave_metadata = slave
    if base_line is None:
        base_line = baseline.Baseline(master, slave)

    interfero.perpendicular_baseline = base_line.perpendicular_baseline(master.number_columns / 2,
                                                                        master.number_rows / 2)
    interfero.temporal_baseline = base_line.temporal_baseline

    interfero._interfero_data = flat_data

    return interfero


def get_geo_points(footprint: footprint.Footprint, pnts_lon: int = 30, pnt_lat: int = 40):
    x = np.linspace(footprint.left(), footprint.right(), pnts_lon)
    y = np.linspace(footprint.top(), footprint.bottom(), pnt_lat)
    xx, yy = np.meshgrid(x, y)

    return np.column_stack((xx.ravel(), yy.ravel()))


def get_geocentric_points(footprint: footprint.Footprint, pnts_lon: int = 30, pnt_lat: int = 40):
    geo_points = get_geo_points(footprint, pnts_lon, pnt_lat)
    geoc = []
    for geo_point in geo_points:
        lon, lat = geo_point
        geoc.append(coordinates.geodetic_to_geocentric(lat, lon))

    return np.array(geoc)


def get_image_coord_satposmaster_satpos_slave(geocentric, master_meta: metadata.MetaData,
                                              slave_meta: metadata.MetaData):
    result = []
    for geoc_point in geocentric:
        master_az_time = master_meta._burst.azimuth_time_from_geocentric(geoc_point)
        satpos_m = master_meta._burst.orbit.interpolate_position(master_az_time)
        slave_az_time = slave_meta._burst.azimuth_time_from_geocentric(geoc_point)
        satpos_s = slave_meta._burst.orbit.interpolate_position(slave_az_time)

        m_x, m_y = master_meta._burst.pixel_from_geocentric(geoc_point)

        result.append((m_x, m_y, geoc_point, satpos_m, satpos_s))

    return result


def get_image_coord_phase(geocentric, master_meta, slave_meta, is_bistatic = False):
    pos = get_image_coord_satposmaster_satpos_slave(geocentric, master_meta, slave_meta)

    result = []
    for m_x, m_y, geoc_point, satpos_m, satpos_s in pos:
        distance_m = np.linalg.norm(geoc_point - satpos_m)
        distance_s = np.linalg.norm(geoc_point - satpos_s)
        if is_bistatic:
            distance_s = (distance_m + distance_s) / 2

        delta_r = distance_s - distance_m
        # Calculate the interferometric phase difference
        delta_phi = (4 * np.pi / master_meta.wavelength) * delta_r

        result.append((m_x, m_y, delta_phi))

    return result


def get_flat_phase_model(pair: resampled_pair.ResampledPair):
    geoc_points = get_geocentric_points(pair.master.metadata.footprint)
    coords_phi = get_image_coord_phase(geoc_points, pair.master.metadata, pair.slave.metadata, pair.bistatic)

    X = np.array([[x, y] for x, y, _ in coords_phi])  # Input features (m_x, m_y)
    y = np.array([phi for _, _, phi in coords_phi])  # Target variable (delta_phi)

    # Create polynomial features up to degree 3
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    # Fit a linear regression model to the polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly


def create_flattened_interferogram(pair: resampled_pair.ResampledPair, phase_model, poly, output = None):
    master = pair.master.slcdata.read()
    slave = pair.slave.slcdata.read()

    # fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    #
    # # Plot the estimated dx shifts
    # # im1 = ax1.imshow(pred_wrapped_phase, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='hsv')
    # im1 = ax1.imshow(np.angle(master * np.conjugate(slave)),
    #                  cmap='hsv')
    #
    # # ax1.set_title('Estimated Topographic Phase')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # fig.colorbar(im1, ax=ax1, label='phase')
    #
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig('/home/timo/Documents/interfero.png', dpi=150)

    flat_interfero = np.zeros(master.shape, dtype=np.complex64)

    xs = np.arange(master.shape[1])
    #ys = np.arange(master.shape[0])

    for y in range(master.shape[0]):
        if output is not None:
            output("Flat interferogram", y, master.shape[0])

        interfero = master[y, :] * np.conjugate(slave[y, :])

        coordinates = np.column_stack((xs, np.full_like(xs, y)))
        point_poly = poly.transform(coordinates)
        pred_phase = phase_model.predict(point_poly)
        flat_phases = np.exp(1j * pred_phase)
        flat_interfero[y, :] = interfero * np.conjugate(flat_phases)

    # fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    #
    # # Plot the estimated dx shifts
    # # im1 = ax1.imshow(pred_wrapped_phase, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='hsv')
    # im1 = ax1.imshow(np.angle(flat_interfero),
    #                  cmap='hsv')
    #
    # #ax1.set_title('Estimated Topographic Phase')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # fig.colorbar(im1, ax=ax1, label='phase')
    #
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig('/home/timo/Documents/flat_interfero.png', dpi=150)

    return flat_interfero


def fromBzarXml(xml_path: str) -> FlatInterferogram:
    interfero = FlatInterferogram()

    root = ET.parse(xml_path).getroot()
    pair_elem = root.find("Interferogram")
    if pair_elem:
        interfero.perpendicular_baseline = float(pair_elem.attrib['baseline'])
        interfero.temporal_baseline = int(pair_elem.attrib['temp_baseline'])
        interfero.master_metadata = metadata.fromBzarXml(pair_elem.find("MasterSlcImage/Band"))
        interfero.slave_metadata = metadata.fromBzarXml(pair_elem.find("SlaveSlcImage/Band"))
        interfero.interferogram_tiff_path = pathlib.Path(xml_path).parent / pair_elem.find("FilePath").text
    return interfero
