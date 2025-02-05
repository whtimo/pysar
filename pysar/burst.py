import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pysar import orbit, footprint
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.constants import c
from rasterio.windows import Window, bounds

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

    def subset(self, window: Window):
        newburst = Burst()
        newburst.orbit = self.orbit
        newburst.range_time_to_first_pixel = self.pixel_to_range_time(window.col_off)
        newburst.first_azimuth_time = self.pixel_to_azimuth_time(window.row_off)
        newburst.column_spacing = self.column_spacing
        newburst.row_spacing = self.row_spacing
        newburst.number_rows = window.height
        newburst.number_columns = window.width
        newburst.first_azimuth_datetime = self.orbit.reference_time + timedelta(seconds=newburst.first_azimuth_time)

        newburst.footprint = self.footprint.subset(window, self.number_columns, self.number_rows)
        return newburst

    def multilook(self, multilook_range=1, multilook_azimuth=1):
        newburst = Burst()
        newburst.orbit = self.orbit
        newburst.range_time_to_first_pixel = self.range_time_to_first_pixel
        newburst.first_azimuth_time = self.first_azimuth_time
        newburst.column_spacing = self.column_spacing * multilook_range
        newburst.row_spacing = self.row_spacing * multilook_azimuth
        newburst.number_rows = int(self.number_rows / multilook_azimuth)
        newburst.number_columns = int(self.number_columns / multilook_range)
        newburst.first_azimuth_datetime = self.first_azimuth_datetime

        newburst.footprint = self.footprint
        return newburst

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

    def pixel_to_range_time(self, pixel):
        return self.range_time_to_first_pixel + pixel * self.column_spacing

    def azimuth_time_to_pixel(self, azimuth_time):
        return (azimuth_time - self.first_azimuth_time) / self.row_spacing

    def pixel_to_azimuth_time(self, pixel):
        return self.first_azimuth_time + pixel * self.row_spacing

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

    def toXml(self, root: ET.Element):
        # Save range_time_to_first_pixel
        if self.range_time_to_first_pixel is not None:
            range_time_elem = ET.SubElement(root, "OneWayTimeToFirstRangePixel")
            range_time_elem.text = str(self.range_time_to_first_pixel)

        # Save first_azimuth_time
        if self.first_azimuth_time is not None:
            azimuth_time_elem = ET.SubElement(root, "TimeOfFirstAzimuthLineFocused")
            azimuth_time_elem.text = str(self.first_azimuth_time)

        # Save column_spacing
        if self.column_spacing is not None:
            column_spacing_elem = ET.SubElement(root, "SlcImageColumnSpacing")
            column_spacing_elem.text = str(self.column_spacing)

        # Save row_spacing
        if self.row_spacing is not None:
            row_spacing_elem = ET.SubElement(root, "SlcImageRowSpacing")
            row_spacing_elem.text = str(self.row_spacing)

        # Save number_rows
        if self.number_rows is not None:
            number_rows_elem = ET.SubElement(root, "NumberOfRows")
            number_rows_elem.text = str(self.number_rows)

        # Save number_columns
        if self.number_columns is not None:
            number_columns_elem = ET.SubElement(root, "NumberOfSamples")
            number_columns_elem.text = str(self.number_columns)

        # Save first_azimuth_datetime
        if self.first_azimuth_datetime is not None:
            azimuth_datetime_elem = ET.SubElement(root, "FirstAzimuthLineDateTime")
            azimuth_datetime_elem.text = self.first_azimuth_datetime.isoformat()


        footprint_elem = ET.SubElement(root, "Footprint")
        self.footprint.toXml(footprint_elem)

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


def fromXml(root : ET.Element, orbit) -> Burst:
    burst = Burst()
    burst.orbit = orbit
    footprint_elem = root.find("Footprint")
    burst.footprint = footprint.fromXml(footprint_elem)
    # Load range_time_to_first_pixel
    range_time_elem = root.find("OneWayTimeToFirstRangePixel")
    if range_time_elem is not None and range_time_elem.text:
        burst.range_time_to_first_pixel = float(range_time_elem.text)

    # Load first_azimuth_time
    azimuth_time_elem = root.find("TimeOfFirstAzimuthLineFocused")
    if azimuth_time_elem is not None and azimuth_time_elem.text:
        burst.first_azimuth_time = float(azimuth_time_elem.text)

    # Load column_spacing
    column_spacing_elem = root.find("SlcImageColumnSpacing")
    if column_spacing_elem is not None and column_spacing_elem.text:
        burst.column_spacing = float(column_spacing_elem.text)

    # Load row_spacing
    row_spacing_elem = root.find("SlcImageRowSpacing")
    if row_spacing_elem is not None and row_spacing_elem.text:
        burst.row_spacing = float(row_spacing_elem.text)

    # Load number_rows
    number_rows_elem = root.find("NumberOfRows")
    if number_rows_elem is not None and number_rows_elem.text:
        burst.number_rows = int(number_rows_elem.text)

    # Load number_columns
    number_columns_elem = root.find("NumberOfSamples")
    if number_columns_elem is not None and number_columns_elem.text:
        burst.number_columns = int(number_columns_elem.text)

    # Load first_azimuth_datetime
    azimuth_datetime_elem = root.find("FirstAzimuthLineDateTime")
    if azimuth_datetime_elem is not None and azimuth_datetime_elem.text:
        burst.first_azimuth_datetime = datetime.fromisoformat(azimuth_datetime_elem.text)


    return burst


def fromBzarXml(root: ET.Element, type: str, orbit, footprint = None) -> Burst:
    burst = Burst()
    burst.orbit = orbit

    # Load range_time_to_first_pixel
    range_time_elem = root.find("OneWayTimeToFirstRangePixel")
    if range_time_elem is not None and range_time_elem.text:
        burst.range_time_to_first_pixel = float(range_time_elem.text)

    # Load first_azimuth_time
    azimuth_time_elem = root.find("TimeOfFirstAzimuthLineFocused")
    if azimuth_time_elem is not None and azimuth_time_elem.text:
        burst.first_azimuth_time = float(azimuth_time_elem.text)

    # Load column_spacing
    column_spacing_elem = root.find("SlcImageColumnSpacing")
    if column_spacing_elem is not None and column_spacing_elem.text:
        burst.column_spacing = float(column_spacing_elem.text)

    if type == 'tsx':
        burst.column_spacing /= 2.0

        # Load row_spacing
    row_spacing_elem = root.find("SlcImageRowSpacing")
    if row_spacing_elem is not None and row_spacing_elem.text:
        burst.row_spacing = float(row_spacing_elem.text)

    # Load number_rows
    number_rows_elem = root.find("NumberOfRows")
    if number_rows_elem is not None and number_rows_elem.text:
        burst.number_rows = int(number_rows_elem.text)

    # Load number_columns
    number_columns_elem = root.find("NumberOfSamples")
    if number_columns_elem is not None and number_columns_elem.text:
        burst.number_columns = int(number_columns_elem.text)

    # Load first_azimuth_datetime
    azimuth_datetime_elem = root.find("AzimuthTime")
    if azimuth_datetime_elem is not None and azimuth_datetime_elem.text:
        burst.first_azimuth_datetime = datetime.fromisoformat(azimuth_datetime_elem.text)

    return burst