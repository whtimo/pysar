import xml.etree.ElementTree as ET
from pysar import footprint
from pysar.sar import orbit, burst
from scipy.constants import c
from scipy.interpolate import interp1d
import numpy as np
import pathlib
import datetime
from rasterio.windows import Window


# Single Burst MetaData
class MetaData:
    """
    Single-Burst SAR Metadata.
    """

    def __init__(self):
        self._burst = None
        self._orbit = None
        self.acquisition_date = None
        self.number_rows = None
        self.number_columns = None
        self.wavelength = None
        self.sensor = None
        self.footprint = None
        self.polarization = None
        self.orbit_direction = None
        #self._incidence_interpolator = None

        # These are not necessary
        self._radar_frequency = None

    def pixel_from_geocentric(self, geocentric: np.array):
        return self._burst.pixel_from_geocentric(geocentric)

    def is_valid(self, x, y, winx=0, winy=0):
        return winx <= x < self.number_columns - winx and winy <= y < self.number_rows - winy

    # def get_incidence_angle(self, col):
    #     return self._incidence_interpolator(col)

    def subset(self, window: Window):
        newmeta = MetaData()

        newmeta._burst = self._burst.subset(window)
        newmeta._orbit = newmeta._burst.orbit
        newmeta.acquisition_date = self.acquisition_date
        newmeta.number_rows = window.height
        newmeta.number_columns = window.width
        newmeta.wavelength = self.wavelength
        newmeta.sensor = self.sensor
        newmeta.footprint = newmeta._burst.footprint
        newmeta.polarization = self.polarization
        newmeta.orbit_direction = self.orbit_direction
        #columns = np.arange(window.col_off, window.col_off + window.width, 100)
        # angles = [self._incidence_interpolator(col) for col in columns]
        # newmeta._incidence_interpolator = interp1d(
        #         columns, angles, kind="cubic", fill_value="extrapolate"
        #     )

        return newmeta

    def multilook(self, multilook_range = 1, multilook_azimuth = 1):
        newmeta = MetaData()

        newmeta._burst = self._burst.multilook(multilook_range, multilook_azimuth)
        newmeta._orbit = newmeta._burst.orbit
        newmeta.acquisition_date = self.acquisition_date
        newmeta.number_rows = int(self.number_rows / multilook_azimuth)
        newmeta.number_columns = int(self.number_columns / multilook_range)
        newmeta.wavelength = self.wavelength
        newmeta.sensor = self.sensor
        newmeta.footprint = newmeta._burst.footprint
        newmeta.polarization = self.polarization
        newmeta.orbit_direction = self.orbit_direction
        # columns = np.arange(0, self.number_columns, 100)
        # angles = [self._incidence_interpolator(col) for col in columns]
        # newcolumns = [col / multilook_range for col in columns]
        # newmeta.incidence_interpolator = interp1d(
        #         newcolumns, angles, kind="cubic", fill_value="extrapolate"
        #     )

        return newmeta

    def toXml(self, root: ET.Element):
        meta_elem = ET.SubElement(root, "MetaData")
        # Save acquisition_date
        if self.acquisition_date is not None:
            acquisition_date_elem = ET.SubElement(meta_elem, "AcquisitionDate")
            acquisition_date_elem.text = self.acquisition_date.isoformat()

        # Save number_rows
        if self.number_rows is not None:
            number_rows_elem = ET.SubElement(meta_elem, "NumberOfRows")
            number_rows_elem.text = str(self.number_rows)

        # Save number_columns
        if self.number_columns is not None:
            number_columns_elem = ET.SubElement(meta_elem, "NumberOfSamples")
            number_columns_elem.text = str(self.number_columns)

        # Save wavelength
        if self.wavelength is not None:
            wavelength_elem = ET.SubElement(meta_elem, "Wavelength")
            wavelength_elem.text = str(self.wavelength)

        # Save sensor
        if self.sensor is not None:
            sensor_elem = ET.SubElement(meta_elem, "SensorName")
            sensor_elem.text = self.sensor

        # Save polarization
        if self.polarization is not None:
            polarization_elem = ET.SubElement(meta_elem, "Polarization")
            polarization_elem.text = self.polarization

        if self.orbit_direction is not None:
            orbitdir_elem = ET.SubElement(meta_elem, "OrbitDirection")
            orbitdir_elem.text = self.orbit_direction

        # # Save incidence_interpolator
        # if self._incidence_interpolator is not None and self.number_columns is not None:
        #     # Sample incidence angles every 500 pixels
        #     columns = np.arange(0, self.number_columns, 500)
        #     incidence_angles = [self._incidence_interpolator(col) for col in columns]
        #
        #     # Save columns and incidence_angles
        #     incidence_elem = ET.SubElement(meta_elem, "IncidenceInterpolator")
        #     for col, angle in zip(columns, incidence_angles):
        #         sample_elem = ET.SubElement(incidence_elem, "sample")
        #         sample_elem.set("column", str(col))
        #         sample_elem.set("angle", str(angle))

        orbit_elem = ET.SubElement(meta_elem, "Orbit")
        self._orbit.toXml(orbit_elem)

        burst_elem = ET.SubElement(meta_elem, "Burst")
        self._burst.toXml(burst_elem)
        footprint_elem = ET.SubElement(meta_elem, "Footprint")
        self.footprint.toXml(footprint_elem)

def fromTSX(xml_path: str, polarization: str) -> MetaData:
    meta = MetaData()
    meta.polarization = polarization
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta._burst = burst.fromTSX(root)
    meta._orbit = meta._burst.orbit
    meta.acquisition_date = meta._burst.first_azimuth_datetime.date()
    meta.number_rows = meta._burst.number_rows
    meta.number_columns = meta._burst.number_columns
    meta.footprint = meta._burst.footprint

    meta._radar_frequency = float(root.find('instrument/radarParameters/centerFrequency').text)
    meta.wavelength = c / meta._radar_frequency
    meta.sensor = root.find('generalHeader/mission').text
    meta.orbit_direction = root.find('productInfo/missionInfo/orbitDirection').text.lower()
    # def parse_grid_points(xml_path):
    #     # Parse the XML string
    #     root = ET.parse(xml_path)
    #
    #     # Extract column and incidence angle values
    #     columns = []
    #     incidence_angles = []
    #
    #     for grid_point in root.findall('.//gridPoint'):
    #         col = int(grid_point.find('col').text)
    #         if not columns.__contains__(col):
    #             inc = float(grid_point.find('inc').text)
    #             columns.append(col)
    #             incidence_angles.append(inc)
    #
    #     return columns, incidence_angles
    #
    # def create_incidence_angle_interpolator(xml_path):
    #     # Parse the grid points
    #     columns, incidence_angles = parse_grid_points(xml_path)
    #
    #     # Sort columns and incidence angles (in case they are not ordered)
    #     sorted_indices = np.argsort(columns)
    #     columns_sorted = np.array(columns)[sorted_indices]
    #     incidence_angles_sorted = np.array(incidence_angles)[sorted_indices]
    #
    #     # Create an interpolation function
    #     interpolator = interp1d(columns_sorted, incidence_angles_sorted, kind='cubic', fill_value='extrapolate')
    #
    #     return interpolator
    #
    # georef_path = pathlib.Path(xml_path).parent / 'ANNOTATION' / 'GEOREF.xml'
    # meta._incidence_interpolator = create_incidence_angle_interpolator(georef_path)

    return meta



def fromXml(root: ET.Element) -> MetaData:
    metadata = MetaData()
    orbit_elem = root.find("Orbit")
    metadata._orbit = orbit.fromXml(orbit_elem)
    burst_elem = root.find("Burst")
    metadata._burst = burst.fromXml(burst_elem, metadata._orbit)
    footprint_elem = root.find("Footprint")
    metadata.footprint = footprint.fromXml(footprint_elem)

    # Load acquisition_date
    acquisition_date_elem = root.find("AcquisitionDate")
    if acquisition_date_elem is not None and acquisition_date_elem.text:
        metadata.acquisition_date = datetime.date.fromisoformat(acquisition_date_elem.text)

    # Load number_rows
    number_rows_elem = root.find("NumberOfRows")
    if number_rows_elem is not None and number_rows_elem.text:
        metadata.number_rows = int(number_rows_elem.text)

    # Load number_columns
    number_columns_elem = root.find("NumberOfSamples")
    if number_columns_elem is not None and number_columns_elem.text:
        metadata.number_columns = int(number_columns_elem.text)

    # Load wavelength
    wavelength_elem = root.find("Wavelength")
    if wavelength_elem is not None and wavelength_elem.text:
        metadata.wavelength = float(wavelength_elem.text)

    # Load sensor
    sensor_elem = root.find("SensorName")
    if sensor_elem is not None and sensor_elem.text:
        metadata.sensor = sensor_elem.text

    orbitdir_elem = root.find("OrbitDirection")
    if orbitdir_elem is not None and orbitdir_elem.text:
        metadata.orbit_direction = orbitdir_elem.text.lower()

    # Load polarization
    polarization_elem = root.find("Polarization")
    if polarization_elem is not None and polarization_elem.text:
        metadata.polarization = polarization_elem.text

    # # Load incidence_interpolator
    # incidence_elem = root.find("IncidenceInterpolator")
    # if incidence_elem is not None:
    #     columns = []
    #     angles = []
    #     for sample_elem in incidence_elem.findall("sample"):
    #         columns.append(float(sample_elem.get("column")))
    #         angles.append(float(sample_elem.get("angle")))
    #
    #     # Reconstruct the interpolator
    #     if columns and angles:
    #         metadata._incidence_interpolator = interp1d(
    #             columns, angles, kind="cubic", fill_value="extrapolate"
    #         )

    return metadata


def fromBzarXml(root: ET.Element) -> MetaData:
    metadata = MetaData()
    meta_elem = root.find("MetaData")
    type = meta_elem.attrib['burst_type']
    ref_time_elem = meta_elem.find("ReferenceTime")
    reference_time = datetime.datetime.fromisoformat(ref_time_elem.text)

    orbit_elem = meta_elem.find("Orbit")
    metadata._orbit = orbit.fromXml(orbit_elem, reference_time)

    burst_elem = meta_elem.find("Bursts/Burst")
    metadata._burst = burst.fromBzarXml(burst_elem, metadata._orbit, type)

    # inc_elem = meta_elem.find("GeoGrid")
    #
    # def parse_grid_points(root: ET.Element):
    #
    #     # Extract column and incidence angle values
    #     columns = []
    #     incidence_angles = []
    #
    #     for grid_point in root.findall('Grid'):
    #         col = int(grid_point.find('Samples').text)
    #         if not columns.__contains__(col):
    #             inc = float(grid_point.find('IncidenceAngle').text)
    #             columns.append(col)
    #             incidence_angles.append(inc)
    #
    #     return columns, incidence_angles
    #
    # def create_incidence_angle_interpolator(root: ET.Element):
    #     # Parse the grid points
    #     columns, incidence_angles = parse_grid_points(root)
    #
    #     # Sort columns and incidence angles (in case they are not ordered)
    #     sorted_indices = np.argsort(columns)
    #     columns_sorted = np.array(columns)[sorted_indices]
    #     incidence_angles_sorted = np.array(incidence_angles)[sorted_indices]
    #
    #     # Create an interpolation function
    #     interpolator = interp1d(columns_sorted, incidence_angles_sorted, kind='cubic', fill_value='extrapolate')
    #
    #     return interpolator
    #
    # metadata._incidence_interpolator = create_incidence_angle_interpolator(inc_elem)

    footprint_elem = meta_elem.find("Footprint")
    metadata.footprint = footprint.fromXml(footprint_elem)
    metadata._burst.footprint = metadata.footprint

    # Load acquisition_date
    acquisition_date_elem = meta_elem.find("AcquisitionDate")
    if acquisition_date_elem is not None and acquisition_date_elem.text:
        metadata.acquisition_date = datetime.date.fromisoformat(acquisition_date_elem.text)

    # Load number_rows
    number_rows_elem = meta_elem.find("NumberOfRows")
    if number_rows_elem is not None and number_rows_elem.text:
        metadata.number_rows = int(number_rows_elem.text)

    # Load number_columns
    number_columns_elem = meta_elem.find("NumberOfSamples")
    if number_columns_elem is not None and number_columns_elem.text:
        metadata.number_columns = int(number_columns_elem.text)

    # Load wavelength
    wavelength_elem = meta_elem.find("Wavelength")
    if wavelength_elem is not None and wavelength_elem.text:
        metadata.wavelength = float(wavelength_elem.text)

    # Load sensor
    sensor_elem = meta_elem.find("SensorName")
    if sensor_elem is not None and sensor_elem.text:
        metadata.sensor = sensor_elem.text

    # Load polarization
    polarization_elem = meta_elem.find("Polarization")
    if polarization_elem is not None and polarization_elem.text:
        metadata.polarization = polarization_elem.text

    orbitdir_elem = meta_elem.find("OrbitDirection")
    if orbitdir_elem is not None and orbitdir_elem.text:
        metadata.orbit_direction = orbitdir_elem.text.lower()

    return metadata

def fromDim(root: ET.Element) -> MetaData:
    metadata = MetaData()

    metadata.footprint = footprint.fromDim(root)
    metadata.sensor = root.find(".//MDATTR[@name='MISSION']").text
    ref_time = datetime.datetime.strptime(root.find(".//MDATTR[@name='STATE_VECTOR_TIME']").text, "%d-%b-%Y %H:%M:%S.%f")
    metadata._orbit = orbit.fromDim(root.find("MDElem[@name='Orbit_State_Vectors']"), ref_time)
    metadata.orbit_direction = root.find(".//MDATTR[@name='PASS']").text
    metadata.polarization = root.find(".//MDATTR[@name='mds1_tx_rx_polar']").text
    metadata._radar_frequency = float(root.find(".//MDATTR[@name='radar_frequency']").text) * 1e6
    metadata.wavelength = c / metadata._radar_frequency
    metadata.number_rows = int(root.find(".//MDATTR[@name='num_output_lines']").text)
    metadata.number_columns = int(root.find(".//MDATTR[@name='num_samples_per_line']").text)
    metadata._burst = burst.fromDim(root, metadata._orbit, metadata.footprint)
    metadata.acquisition_date = metadata._burst.first_azimuth_datetime.date()

    return metadata