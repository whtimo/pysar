import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pysar import footprint
from pysar.sar import orbit
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.constants import c
from rasterio.windows import Window
from multiprocessing import Pool

# Define the objective function (outside the loop to avoid redefinition)
def __objective(time, target_pos, splx, sply, splz):
    orbit_pos = np.array([splx(time), sply(time), splz(time)])
    return np.sum((orbit_pos - target_pos) ** 2)

# Worker function for a single (x, y, z) tuple
def _worker(args):
    target_p, splx, sply, splz, time_min, time_max, time_tol = args
    result = minimize_scalar(
        __objective,
        bounds=(time_min, time_max),
        method='bounded',
        args=(target_p, splx, sply, splz,),
        options={'xatol': time_tol}
    )
    return result.x if result.success else np.nan


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
        self.azimuth_timing_shift = 0
        self.range_delay = 0

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
    def azimuth_time_from_geocentric_single(self, target):
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

    def azimuth_time_from_geocentric_singletask_old(self, target):
        """
        Finds the time within [time_min, time_max] that minimizes the distance
        between the target position and the orbit's position.

        Args:
            target: Target coordinates as a tuple of 3 ndarrays (x, y, z), each of shape (m, n).

        Returns:
            np.ndarray: A 2D array of optimal times, one for each (x, y, z) tuple in the input grids.
                        Returns None if the optimization fails for any coordinate.
        """

        if not hasattr(target[0], '__len__'):
            return self.azimuth_time_from_geocentric_single(target)

        # Unpack the target coordinates
        x_grid, y_grid, z_grid = target

        # Ensure the input grids have the same shape
        if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
            raise ValueError("Input grids (x, y, z) must have the same shape.")

        # Get the time bounds and tolerance
        time_min = self.orbit.times[0]
        time_max = self.orbit.times[-1]
        time_tol = 1e-11

        # Initialize the output grid for optimal times
        optimal_times = np.zeros_like(x_grid, dtype=float)

        # Iterate over each (x, y, z) tuple in the grids
        for i in range(x_grid.shape[0]):
            print(f'{i} / {x_grid.shape[0]}')

            for j in range(x_grid.shape[1]):
                # Target position as a 1D array
                target_pos = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])

                # Define the objective function for this target
                def objective(time):
                    orbit_pos = self.orbit.interpolate_position(time)
                    return np.sum((orbit_pos - target_pos) ** 2)

                # Perform the optimization
                result = minimize_scalar(
                    objective,
                    bounds=(time_min, time_max),
                    method='bounded',
                    options={'xatol': time_tol}
                )

                # Store the result if successful
                if result.success:
                    optimal_times[i, j] = result.x
                else:
                    # If optimization fails, return None for the entire grid
                    return None

        return optimal_times

    def azimuth_time_from_geocentric_singletask(self, target):
        """
        Finds the time within [time_min, time_max] that minimizes the distance
        between the target position and the orbit's position.

        Args:
            target: Target coordinates as a tuple of 3 ndarrays (x, y, z), each of shape (m, n).

        Returns:
            np.ndarray: A 2D array of optimal times, one for each (x, y, z) tuple in the input grids.
                        Returns None if the optimization fails for any coordinate.
        """

        if not hasattr(target[0], '__len__'):
            return self.azimuth_time_from_geocentric_single(target)

        # Unpack the target coordinates
        x_grid, y_grid, z_grid = target

        x_grid = np.atleast_2d(x_grid)
        y_grid = np.atleast_2d(y_grid)
        z_grid = np.atleast_2d(z_grid)

        # Ensure the input grids have the same shape
        if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
            raise ValueError("Input grids (x, y, z) must have the same shape.")

        # Get the time bounds and tolerance
        time_min = self.orbit.times[0]
        time_max = self.orbit.times[-1]
        time_tol = 1e-11

        # Initialize the output grid for optimal times
        optimal_times = np.zeros_like(x_grid, dtype=float)

        # Iterate over each (x, y, z) tuple in the grids
        for i in range(x_grid.shape[0]):
            print(f'{i} / {x_grid.shape[0]}')

            for j in range(x_grid.shape[1]):
                # Target position as a 1D array
                target_pos = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])

                # Define the objective function for this target
                def objective(time):
                    orbit_pos = self.orbit.interpolate_position(time)
                    return np.sum((orbit_pos - target_pos) ** 2)

                # Perform the optimization
                result = minimize_scalar(
                    objective,
                    bounds=(time_min, time_max),
                    method='bounded',
                    options={'xatol': time_tol}
                )

                # Store the result if successful
                if result.success:
                    optimal_times[i, j] = result.x
                else:
                    # If optimization fails, return None for the entire grid
                    return None

        return optimal_times #.squeeze(axis=0)

    def azimuth_time_from_geocentric(self, target):
        """
        Finds the time within [time_min, time_max] that minimizes the distance
        between the target position and the orbit's position.

        Args:
            target: Target coordinates as a tuple of 3 ndarrays (x, y, z), each of shape (m, n).

        Returns:
            np.ndarray: A 2D array of optimal times, one for each (x, y, z) tuple in the input grids.
                        Returns None if the optimization fails for any coordinate.
        """
        if not hasattr(target[0], '__len__') or len(target[0]) < 100:
            return self.azimuth_time_from_geocentric_singletask(target)

        # Unpack the target coordinates
        x_grid, y_grid, z_grid = target

        # Validate input grids
        if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
            raise ValueError("Input grids (x, y, z) must have the same shape.")

        # Extract time bounds and tolerance
        time_min = self.orbit.times[0]
        time_max = self.orbit.times[-1]
        time_tol = 1e-11

        # Flatten the grids for parallel processing
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        z_flat = z_grid.ravel()
        total_points = len(x_flat)


        args_list = []
        for i in range(total_points):
            target_pos = np.array([x_flat[i], y_flat[i], z_flat[i]])
            args_list.append((target_pos, self.orbit._spline_x, self.orbit._spline_y, self.orbit._spline_z, time_min, time_max, time_tol))

        target_p, splx, sply, splz, time_min, time_max, time_tol = args_list[0]
        # Parallelize using Pool
        with Pool() as pool:
            # Map indices to the worker function
            optimal_times_flat = pool.map(_worker, args_list)

        # Reshape the results into a 2D grid
        optimal_times = np.array(optimal_times_flat).reshape(x_grid.shape)

        # Check for failures (NaNs)
        if np.isnan(optimal_times).any():
            return None

        return optimal_times


    def range_time_to_pixel(self, range_time):
        return (range_time + self.range_delay - self.range_time_to_first_pixel) / self.column_spacing

    def pixel_to_range_time(self, pixel):
        return (self.range_time_to_first_pixel +
                pixel * self.column_spacing) - self.range_delay

    def azimuth_time_to_pixel(self, azimuth_time):
        azitime = azimuth_time + self.azimuth_timing_shift
        return (azitime - self.first_azimuth_time) / self.row_spacing

    def pixel_to_azimuth_time(self, pixel):
        azitime = self.first_azimuth_time + pixel * self.row_spacing
        return azitime - self.azimuth_timing_shift

    # Returns the pixel possition for a given geocentric coordinate
    def pixel_from_geocentric(self, geocentric, allow_parallel=True):
        if allow_parallel:
            az_time = self.azimuth_time_from_geocentric(geocentric)
        else:
            az_time = self.azimuth_time_from_geocentric_singletask(geocentric)

        if az_time is None: return None
        satpos = self.orbit.interpolate_position(az_time)
        distance = np.linalg.norm(geocentric - satpos, axis=0)
        rg_time = distance / c
        x = self.range_time_to_pixel(rg_time)
        y = self.azimuth_time_to_pixel(az_time)
        return [x, y]

    def is_valid(self, x, y, winx = 0, winy = 0):
        return winx <= x < self.number_columns-winx and winy <= y < self.number_rows - winy

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

        azimuth_delay_elem = ET.SubElement(root, "AzimuthTimingShift")
        azimuth_delay_elem.text = str(self.azimuth_timing_shift)

        range_delay_elem = ET.SubElement(root, "RangeAtmosphericDelay")
        range_delay_elem.text = str(self.range_delay)

        footprint_elem = ET.SubElement(root, "Footprint")
        self.footprint.toXml(footprint_elem)

def fromTSX(root: ET.Element, georef_path:str = None) -> Burst:
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

    if georef_path is not None:
        georef_tree = ET.parse(georef_path)
        geo_reference = georef_tree.getroot()

        signal_propagation_effects = geo_reference.find('signalPropagationEffects')

        iono = 0
        atmo = 0
        for range_delay in signal_propagation_effects.findall('rangeDelay'):
            source = range_delay.get('source')
            if source == "IONO":
                iono = float(range_delay.find('coefficient').text)
            elif source == "ATMOS":
                atmo = float(range_delay.find('coefficient').text)

        burst.range_delay = (iono + atmo) / 2.0
        azimuth_shift = signal_propagation_effects.find('azimuthShift')
        burst.azimuth_timing_shift = float(azimuth_shift.find('coefficient').text)

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

    azimuth_delay_elem = root.find("AzimuthTimingShift")
    if azimuth_delay_elem is not None:
        burst.azimuth_delay = float(azimuth_delay_elem.text)

    range_delay_elem = root.find("RangeAtmosphericDelay")
    if range_delay_elem is not None:
        burst.range_delay = float(range_delay_elem.text)

    return burst


def fromBzarXml(root: ET.Element, orbit, type: str, footprint = None) -> Burst:
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

    azimuth_delay_elem = root.find("AzimuthTimingShift")
    if azimuth_delay_elem is not None:
        burst.azimuth_delay = float(azimuth_delay_elem.text)

    range_iono_delay_elem = root.find("TsxMetaRangeDelayIonosphere")
    range_tropo_delay_elem = root.find("TsxMetaRangeDelayTroposphere")
    if range_iono_delay_elem is not None and range_tropo_delay_elem is not None:
        iono_delay = float(range_tropo_delay_elem.text)
        tropo_delay = float(range_tropo_delay_elem.text)
        burst.range_delay = (iono_delay + tropo_delay) / 2.0

    return burst

def fromDim(root: ET.Element, orbit, footprint) -> Burst:
    burst = Burst()
    burst.orbit = orbit
    burst.footprint = footprint
    slant_range_to_first_pixel = float(root.find(".//MDATTR[@name='slant_range_to_first_pixel']").text)
    burst.range_time_to_first_pixel = slant_range_to_first_pixel / c
    burst._range_sampling_rate =  float(root.find(".//MDATTR[@name='range_sampling_rate']").text) * 1e6
    burst.column_spacing = 1 / burst._range_sampling_rate
    burst.row_spacing = float(root.find(".//MDATTR[@name='line_time_interval']").text)
    burst.number_rows = int(root.find(".//MDATTR[@name='num_output_lines']").text)
    burst.number_columns = int(root.find(".//MDATTR[@name='num_samples_per_line']").text)
    first_line_time_str = root.find(".//MDATTR[@name='first_line_time']").text
    burst.first_azimuth_datetime = datetime.strptime(first_line_time_str, "%d-%b-%Y %H:%M:%S.%f")
    rt = burst.orbit.reference_time
    t = (burst.first_azimuth_datetime - rt).total_seconds()
    burst.first_azimuth_time = t

    return burst