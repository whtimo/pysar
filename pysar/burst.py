import xml.etree.ElementTree as ET
from datetime import datetime
from pysar import orbit, footprint
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.constants import c


class Burst:
    def __init__(self):
        self.orbit = None
        self.range_time_to_first_pixel = None  # First pixel value as a float
        self.first_azimuth_time = None  # First azimuth line in seconds with refrence to reference time of orbit
        self.column_spacing = None  # Fast time in seconds
        self.row_spacing = None  # Slow-time in seconds
        self.number_rows = None
        self.number_columns = None
        self.first_azimuth_datetime = None
        self.footprint = None

        # These are not necessary
        self._prf = None
        self._total_bandwidth_range = None

    # Returns the azimuth time and the satellite position for the given time that are closest to the given geocentric position
    def azimuth_time_from_geocentric(self, target: np.array):
        """
        Finds the time within [time_min, time_max] that minimizes the distance
        between the target position and the orbit's position.

        Args:
            target: Target coordinate [x, y, z] as np.array.

        Returns:
            float: The optimal time that gives the closest position.
            None: If the optimization fails to converge.
        """

        # time_min (float): Lower bound of the time interval.
        # time_max (float): Upper bound of the time interval.
        # time_tol (float): Tolerance for the time optimization. Differences in
        # time smaller than this are considered negligible.
        time_min = self.orbit.times[0]
        time_max = self.orbit.times[-1]
        time_tol = 1e-11

        def objective(time):
            orbit_pos = self.orbit.interpolate_position(time)
            return np.sum((orbit_pos - target) ** 2)

        result = minimize_scalar(
            objective,
            bounds=(time_min, time_max),
            method='bounded',
            options={'xatol': time_tol}
        )

        if result.success:
            return result.x
        else:
            return None

    def range_time_to_pixel(self, range_time):
        return (range_time - self.range_time_to_first_pixel) / self.column_spacing

    def azimuth_time_to_pixel(self, azimuth_time):
        return (azimuth_time - self.first_azimuth_time) / self.row_spacing

    # Returns the pixel possition for a given geocentric coordinate
    def pixel_from_geocentric(self, geocentric: np.array):
        az_time = self.azimuth_time_from_geocentric(geocentric)
        if az_time is None: return None
        satpos = self.orbit.interpolate_position(az_time)
        distance = np.linalg.norm(geocentric - satpos)
        rg_time = distance / c
        x = self.range_time_to_pixel(rg_time)
        y = self.azimuth_time_to_pixel(az_time)
        return [x, y]


def fromTSX(root: ET.Element) -> Burst:
    burst = Burst()

    # Find the <sceneInfo> tag
    product_info = root.find('productInfo')
    scene_info = product_info.find('sceneInfo')
    if scene_info is None:
        raise ValueError("No <sceneInfo> tag found in the XML document.")

    # Extract the start time (<start/timeUTC>)
    start_time_utc = scene_info.find('start/timeUTC')
    if start_time_utc is not None:
        time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        burst.first_azimuth_datetime = datetime.strptime(start_time_utc.text, time_format)

    # Extract the first pixel value (<rangeTime/firstPixel>)
    first_pixel = scene_info.find('rangeTime/firstPixel')
    if first_pixel is not None:
        burst.range_time_to_first_pixel = float(first_pixel.text) / 2.0

    imageDataInfo = product_info.find('imageDataInfo')
    imageRaster = imageDataInfo.find('imageRaster')
    burst.number_rows = int(imageRaster.find('numberOfRows').text)
    burst.number_columns = int(imageRaster.find('numberOfColumns').text)
    burst.column_spacing = float(imageRaster.find('rowSpacing').text) / 2.0
    burst.row_spacing = float(imageRaster.find('columnSpacing').text)

    burst._prf = float(root.find('productSpecific/complexImageInfo/commonPRF').text)
    burst._total_bandwidth_range = float(root.find('processing/processingParameter/totalProcessedRangeBandwidth').text)

    burst.orbit = orbit.fromTSX(root)
    burst.first_azimuth_time = burst.orbit.seconds_from_reference_time(burst.first_azimuth_datetime)
    burst.footprint = footprint.fromTSX(root)
    return burst
