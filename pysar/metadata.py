import xml.etree.ElementTree as ET
from pysar import burst, footprint
from scipy.constants import c
from scipy.interpolate import interp1d
import numpy as np
import pathlib
import datetime
from rasterio.windows import Window, bounds

# Single Burst MetaData
class MetaData:
    def __init__(self):
        self.burst = None
        self.acquisition_date = None
        self.number_rows = None
        self.number_columns = None
        self.wavelength = None
        self.sensor = None
        self.footprint = None
        self.polarization = None
        self.incidence_interpolator = None

        # These are not necessary
        self._radar_frequency = None

    def get_incidence_angle(self, col):
        return self.incidence_interpolator(col)

    def subset(self, window: Window):
        newmeta = MetaData()

        newmeta.burst = self.burst.subset(window)
        newmeta.acquisition_date = self.acquisition_date
        newmeta.number_rows = window.height
        newmeta.number_columns = window.width
        newmeta.wavelength = self.wavelength
        newmeta.sensor = self.sensor
        newmeta.footprint = newmeta.burst.footprint
        newmeta.polarization = self.polarization
        columns = np.arange(window.col_off, window.col_off + window.width, 100)
        angles = [self.incidence_interpolator(col) for col in columns]
        newmeta.incidence_interpolator = interp1d(
                columns, angles, kind="cubic", fill_value="extrapolate"
            )

        return newmeta

    def multilook(self, multilook_range = 1, multilook_azimuth = 1):
        newmeta = MetaData()

        newmeta.burst = self.burst.multilook(multilook_range, multilook_azimuth)
        newmeta.acquisition_date = self.acquisition_date
        newmeta.number_rows = int(self.number_rows / multilook_azimuth)
        newmeta.number_columns = int(self.number_columns / multilook_range)
        newmeta.wavelength = self.wavelength
        newmeta.sensor = self.sensor
        newmeta.footprint = newmeta.burst.footprint
        newmeta.polarization = self.polarization
        columns = np.arange(0, self.number_columns, 100)
        angles = [self.incidence_interpolator(col) for col in columns]
        newcolumns = [col / multilook_range for col in columns]
        newmeta.incidence_interpolator = interp1d(
                newcolumns, angles, kind="cubic", fill_value="extrapolate"
            )

        return newmeta

    def toXml(self, root: ET.Element):
        # Save acquisition_date
        if self.acquisition_date is not None:
            acquisition_date_elem = ET.SubElement(root, "acquisition_date")
            acquisition_date_elem.text = self.acquisition_date.isoformat()

        # Save number_rows
        if self.number_rows is not None:
            number_rows_elem = ET.SubElement(root, "number_rows")
            number_rows_elem.text = str(self.number_rows)

        # Save number_columns
        if self.number_columns is not None:
            number_columns_elem = ET.SubElement(root, "number_columns")
            number_columns_elem.text = str(self.number_columns)

        # Save wavelength
        if self.wavelength is not None:
            wavelength_elem = ET.SubElement(root, "wavelength")
            wavelength_elem.text = str(self.wavelength)

        # Save sensor
        if self.sensor is not None:
            sensor_elem = ET.SubElement(root, "sensor")
            sensor_elem.text = self.sensor

        # Save polarization
        if self.polarization is not None:
            polarization_elem = ET.SubElement(root, "polarization")
            polarization_elem.text = self.polarization

        # Save incidence_interpolator
        if self.incidence_interpolator is not None and self.number_columns is not None:
            # Sample incidence angles every 500 pixels
            columns = np.arange(0, self.number_columns, 500)
            incidence_angles = [self.incidence_interpolator(col) for col in columns]

            # Save columns and incidence_angles
            incidence_elem = ET.SubElement(root, "incidence_interpolator")
            for col, angle in zip(columns, incidence_angles):
                sample_elem = ET.SubElement(incidence_elem, "sample")
                sample_elem.set("column", str(col))
                sample_elem.set("angle", str(angle))

        burst_elem = ET.SubElement(root, "burst")
        self.burst.toXml(burst_elem)
        footprint_elem = ET.SubElement(root, "footprint")
        self.footprint.toXml(footprint_elem)

def fromTSX(xml_path: str, polarization: str) -> MetaData:
    meta = MetaData()
    meta.polarization = polarization
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta.burst = burst.fromTSX(root)
    meta.acquisition_date = meta.burst.first_azimuth_datetime.date()
    meta.number_rows = meta.burst.number_rows
    meta.number_columns = meta.burst.number_columns
    meta.footprint = meta.burst.footprint

    meta._radar_frequency = float(root.find('instrument/radarParameters/centerFrequency').text)
    meta.wavelength = c / meta._radar_frequency
    meta.sensor = root.find('generalHeader/mission').text

    def parse_grid_points(xml_path):
        # Parse the XML string
        root = ET.parse(xml_path)

        # Extract column and incidence angle values
        columns = []
        incidence_angles = []

        for grid_point in root.findall('.//gridPoint'):
            col = int(grid_point.find('col').text)
            if not columns.__contains__(col):
                inc = float(grid_point.find('inc').text)
                columns.append(col)
                incidence_angles.append(inc)

        return columns, incidence_angles

    def create_incidence_angle_interpolator(xml_path):
        # Parse the grid points
        columns, incidence_angles = parse_grid_points(xml_path)

        # Sort columns and incidence angles (in case they are not ordered)
        sorted_indices = np.argsort(columns)
        columns_sorted = np.array(columns)[sorted_indices]
        incidence_angles_sorted = np.array(incidence_angles)[sorted_indices]

        # Create an interpolation function
        interpolator = interp1d(columns_sorted, incidence_angles_sorted, kind='cubic', fill_value='extrapolate')

        return interpolator

    georef_path = pathlib.Path(xml_path).parent / 'ANNOTATION' / 'GEOREF.xml'
    meta.incidence_interpolator = create_incidence_angle_interpolator(georef_path)
    return meta



def fromXml(root: ET.Element) -> MetaData:
    metadata = MetaData()
    burst_elem = root.find("burst")
    metadata.burst = burst.fromXml(burst_elem)
    footprint_elem = root.find("footprint")
    metadata.footprint = footprint.fromXml(footprint_elem)

    # Load acquisition_date
    acquisition_date_elem = root.find("acquisition_date")
    if acquisition_date_elem is not None and acquisition_date_elem.text:
        metadata.acquisition_date = datetime.date.fromisoformat(acquisition_date_elem.text)

    # Load number_rows
    number_rows_elem = root.find("number_rows")
    if number_rows_elem is not None and number_rows_elem.text:
        metadata.number_rows = int(number_rows_elem.text)

    # Load number_columns
    number_columns_elem = root.find("number_columns")
    if number_columns_elem is not None and number_columns_elem.text:
        metadata.number_columns = int(number_columns_elem.text)

    # Load wavelength
    wavelength_elem = root.find("wavelength")
    if wavelength_elem is not None and wavelength_elem.text:
        metadata.wavelength = float(wavelength_elem.text)

    # Load sensor
    sensor_elem = root.find("sensor")
    if sensor_elem is not None and sensor_elem.text:
        metadata.sensor = sensor_elem.text

    # Load polarization
    polarization_elem = root.find("polarization")
    if polarization_elem is not None and polarization_elem.text:
        metadata.polarization = polarization_elem.text

    # Load incidence_interpolator
    incidence_elem = root.find("incidence_interpolator")
    if incidence_elem is not None:
        columns = []
        angles = []
        for sample_elem in incidence_elem.findall("sample"):
            columns.append(float(sample_elem.get("column")))
            angles.append(float(sample_elem.get("angle")))

        # Reconstruct the interpolator
        if columns and angles:
            metadata.incidence_interpolator = interp1d(
                columns, angles, kind="cubic", fill_value="extrapolate"
            )

    return metadata