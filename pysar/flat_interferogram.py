from pysar import metadata, footprint, resampled_pair, coordinates, cpl_float_memory_slcdata,baseline
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from rasterio.windows import Window
import rasterio
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pathlib

class FlatInterferogram:
    def __init__(self, filepath:str = None):
        self.master_metadata = None
        self.slave_metadata = None
        self.perpendicular_baseline = None
        self.temporal_baseline = None
        self.interferogram_tiff_path = None
        self.data = None

        if filepath:
            root = ET.parse(filepath).getroot()
            pair_elem = root.find("Interferogram")
            if pair_elem:
                self.perpendicular_baseline =  float(pair_elem.attrib['perpendicular_baseline'])
                self.temporal_baseline = int(pair_elem.attrib['temporal_baseline'])
                self.master_metadata = metadata.fromXml(pair_elem.find("Master/MetaData"))
                self.slave_metadata = metadata.fromXml(pair_elem.find("Slave/MetaData"))

                self.interferogram_tiff_path = pathlib.Path(filepath).parent / pair_elem.find("FilePath").text

    def __getTiffName(self, path, overwrite: bool = True):
        counter = 0
        tiff_name = f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.flat.interfero.tiff'
        while (pathlib.Path(path) / tiff_name).exists() and not overwrite:
            counter += 1
            tiff_name = f'{self.master_metadata.sensor}_{counter}_{self.master_metadata.acquisition_date.isoformat()}__{self.slave_metadata.sensor}_{self.slave_metadata.acquisition_date.isoformat()}.flat.interfero.tiff'

        return tiff_name

    def save(self, filepath:str):
        root = ET.Element("PySar")
        pair_elem = ET.SubElement(root, "Interferogram")
        pair_elem.attrib["perpendicular_baseline"] = str(self.perpendicular_baseline)
        pair_elem.attrib["temporal_baseline"] = str(self.temporal_baseline)
        master_elem = ET.SubElement(pair_elem, "Master")
        self.master_metadata.toXml(master_elem)
        slave_elem = ET.SubElement(pair_elem, "Slave")
        self.slave_metadata.toXml(slave_elem)
        tiff_name = self.__getTiffName(filepath, True)
        self.__openfile()

        metadata = {
            "driver": "GTiff",  # GeoTIFF format
            "height": self.data.shape[0],  # Number of rows
            "width": self.data.shape[1],  # Number of columns
            "count": 1,  # Single band (complex data)
            "dtype": np.complex64,  # Complex float32 data type
            "transform": rasterio.Affine.identity(),  # Identity transform (no georeferencing)
        }

        with rasterio.open(pathlib.Path(filepath).parent / tiff_name, "w", **metadata) as dst:
            dst.write(self.data, 1)  # Write the complex array to the first band

        file_path_elem = ET.SubElement(pair_elem, "FilePath")
        file_path_elem.text = str(tiff_name)
        xml_str = ET.tostring(root, encoding="utf-8")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        # Write the pretty-printed XML to a file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(pretty_xml)


    def __openfile(self):
        if self.data is None:
            if self.interferogram_tiff_path is not None and len(self.interferogram_tiff_path) > 0:
                self.data = rasterio.open(self.interferogram_tiff_path)

    def getWidth(self) -> int:
        self.__openfile()
        return self.data.width

    def getHeight(self) -> int:
        self.__openfile()
        return self.data.height

    def read(self, window: Window = None) -> np.ndarray:
        self.__openfile()
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

def createFlatInterferogram(master: metadata.MetaData, slave: metadata.MetaData, flat_data: np.ndarray, base_line: baseline.Baseline = None):
    interfero = FlatInterferogram()
    interfero.master_metadata = master
    interfero.slave_metadata = slave
    if base_line is None:
        base_line = baseline.Baseline(master, slave)

    interfero.perpendicular_baseline = base_line.perpendicular_baseline(master.number_columns / 2, master.number_rows / 2)
    interfero.temporal_baseline = base_line.temporal_baseline

    interfero.data = flat_data

    return interfero


def get_geo_points(footprint: footprint.Footprint, pnts_lon:int = 30, pnt_lat:int = 40):
    x = np.linspace(footprint.left(), footprint.right(), pnts_lon)
    y = np.linspace(footprint.top(), footprint.bottom(), pnt_lat)
    xx, yy = np.meshgrid(x, y)

    return np.column_stack((xx.ravel(), yy.ravel()))

def get_geocentric_points(footprint: footprint.Footprint, pnts_lon:int = 30, pnt_lat:int = 40):
    geo_points = get_geo_points(footprint, pnts_lon, pnt_lat)
    geoc = []
    for geo_point in geo_points:
        lon, lat = geo_point
        geoc.append(coordinates.geodetic_to_geocentric(lat, lon))

    return np.array(geoc)

def get_image_coord_satposmaster_satpos_slave(geocentric, master_meta: metadata.MetaData, slave_meta: metadata.MetaData):

    result = []
    for geoc_point in geocentric:
        master_az_time = master_meta.burst.azimuth_time_from_geocentric(geoc_point)
        satpos_m = master_meta.burst.orbit.interpolate_position(master_az_time)
        slave_az_time = slave_meta.burst.azimuth_time_from_geocentric(geoc_point)
        satpos_s = slave_meta.burst.orbit.interpolate_position(slave_az_time)

        m_x, m_y = master_meta.burst.pixel_from_geocentric(geoc_point)

        result.append((m_x, m_y, geoc_point, satpos_m, satpos_s))

    return result

def get_image_coord_phase(geocentric, master_meta, slave_meta):
    pos = get_image_coord_satposmaster_satpos_slave(geocentric, master_meta, slave_meta)

    result = []
    for m_x, m_y, geoc_point, satpos_m, satpos_s in pos:
        distance_m = np.linalg.norm(geoc_point - satpos_m)
        distance_s = np.linalg.norm(geoc_point - satpos_s)
        delta_r = distance_s - distance_m
        # Calculate the interferometric phase difference
        delta_phi = (4 * np.pi / master_meta.wavelength) * delta_r

        result.append((m_x, m_y, delta_phi))

    return result


def get_flat_phase_model(pair: resampled_pair.ResampledPair):
    geoc_points = get_geocentric_points(pair.master.metadata.footprint)
    coords_phi = get_image_coord_phase(geoc_points, pair.master.metadata, pair.slave.metadata)

    X = np.array([[x, y] for x, y, _ in coords_phi])  # Input features (m_x, m_y)
    y = np.array([phi for _, _, phi in coords_phi])  # Target variable (delta_phi)

    # Create polynomial features up to degree 3
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    # Fit a linear regression model to the polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly


def create_flattened_interferogram(pair: resampled_pair.ResampledPair, phase_model, poly):
    master = pair.master.slcdata.read()
    slave = pair.slave.slcdata.read()

    flat_interfero = np.zeros(master.shape, dtype=np.complex64)

    xs = np.arange(master.shape[1])
    #ys = np.arange(master.shape[0])

    for y in range(master.shape[0]):
        print(f'Flat Interferogram: {y} / {master.shape[0]}')
        interfero = master[y,:] * np.conjugate(slave[y,:])
        coordinates = np.column_stack((xs, np.full_like(xs, y)))
        point_poly = poly.transform(coordinates)
        pred_phase = phase_model.predict(point_poly)
        flat_phases = np.exp(1j * pred_phase)
        flat_interfero[y,:] = interfero * np.conjugate(flat_phases)

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
        interfero.interferogram_tiff_path = pathlib.Path(xml_path).parent /  pair_elem.find("FilePath").text
    return interfero

def createFilename(master: metadata.MetaData, slave: metadata.MetaData, directory:str) -> pathlib.Path:
    xml_path = pathlib.Path(directory) / f'{master.sensor}_{master.acquisition_date.isoformat()}__{slave.sensor}_{slave.acquisition_date.isoformat()}.pysar.flat.interfero.xml'
    return xml_path