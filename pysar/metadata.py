import xml.etree.ElementTree as ET
from pysar import burst
from scipy.constants import c
from scipy.interpolate import interp1d
import numpy as np
import pathlib


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
