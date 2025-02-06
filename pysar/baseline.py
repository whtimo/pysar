import numpy as np
from pysar import footprint, coordinates, burst, metadata
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def get_random_points(count: int, footprint: footprint.Footprint, min_height: float, max_height: float):

    result = []
    for _ in range(count):
        lon = random.uniform(footprint.left(), footprint.right())
        lat = random.uniform(footprint.top(), footprint.bottom())
        height = random.uniform(min_height, max_height)
        result.append((lat, lon, height))

    return result

def get_satellite_position(geocentric: np.array, burst: burst.Burst) -> np.array:

    az_time = burst.azimuth_time_from_geocentric(geocentric)
    satpos = burst.orbit.interpolate_position(az_time)

    return satpos


def calculate_baselines(ground_pos: np.array, master_pos: np.array, slave_pos: np.array):
    """
    Calculate the baseline, parallel baseline, and perpendicular baseline.

    Parameters:
    ground_pos (np.array): Ground position in geocentric coordinates (X, Y, Z).
    master_pos (np.array): Master satellite position in geocentric coordinates (X, Y, Z).
    slave_pos (np.array): Slave satellite position in geocentric coordinates (X, Y, Z).

    Returns:
    baseline (float): Total baseline magnitude.
    B_parallel (float): Parallel baseline.
    B_perpendicular (float): Perpendicular baseline.
    """
    # Calculate the baseline vector
    baseline_vector = slave_pos - master_pos

    # Calculate the line of sight (LOS) vector from master satellite to ground position
    los_vector = ground_pos - master_pos

    # Normalize the LOS vector
    los_vector_unit = los_vector / np.linalg.norm(los_vector)

    # Calculate the parallel baseline (projection of baseline_vector onto LOS)
    B_parallel = np.dot(baseline_vector, los_vector_unit)

    # Calculate the perpendicular baseline
    B_perpendicular_vector = baseline_vector - B_parallel * los_vector_unit
    B_perpendicular = np.linalg.norm(B_perpendicular_vector)

    # Calculate the total baseline magnitude
    baseline = np.linalg.norm(baseline_vector)

    return baseline, B_parallel, B_perpendicular


def calculate_incidence_angle(ground_pos, satellite_pos):
    """
    Calculate the incidence angle for a given ground point and satellite position.

    Parameters:
    ground_pos (np.array): Ground position in geocentric coordinates (X, Y, Z).
    satellite_pos (np.array): Satellite position in geocentric coordinates (X, Y, Z).

    Returns:
    incidence_angle (float): Incidence angle in radians. (Use np.degrees if you need degrees)
    """
    # Calculate the line of sight (LOS) vector from the satellite to the ground point
    los_vector = ground_pos - satellite_pos

    # Normalize the LOS vector
    los_vector_unit = los_vector / np.linalg.norm(los_vector)

    # The local vertical is the ground position vector (normal to the Earth's surface)
    local_vertical_unit = ground_pos / np.linalg.norm(ground_pos)

    # Calculate the incidence angle using the dot product
    cos_incidence = np.dot(los_vector_unit, local_vertical_unit)
    incidence_angle = np.arccos(cos_incidence)

    return incidence_angle

def get_perpendicular_baseline_estimator(primary_burst: burst.Burst, secondary_burst: burst.Burst, points=1000, min_height=0.0, max_height=2000.0, degree=3):
    """
        Train a polynomial regression model to estimate the perpendicular baseline from (x, y, height).

        Parameters:
        primary_burst (burst.Burst): Primary burst (master).
        secondary_burst (burst.Burst): Secondary burst (slave).
        points (int): Number of sample points for training.
        min_height (float): Minimum height for sampling.
        max_height (float): Maximum height for sampling.
        degree (int): Degree of the polynomial regression (default=3).

        Returns:
        model: Trained scikit-learn pipeline (polynomial features + linear regression).

        Usage: predicted_baseline = model.predict([[x_new, y_new, height_new]])
    """
    pnts = get_random_points(points*2, primary_burst.footprint, min_height, max_height)

    # Collect features (x, y, height) and targets (perpendicular baseline)
    X = []
    y = []
    for lat, lon, h in pnts:
        geocentric = coordinates.geodetic_to_geocentric(lat, lon, h)
        x_pixel, y_pixel = primary_burst.pixel_from_geocentric(geocentric)
        if primary_burst.is_valid(x_pixel, y_pixel):
            primary_pos = get_satellite_position(geocentric, primary_burst)
            seondary_pos = get_satellite_position(geocentric, secondary_burst)
            _, _, perp_baseline = calculate_baselines(geocentric, primary_pos, seondary_pos)

            X.append([x_pixel, y_pixel, h])
            y.append(perp_baseline)
            if len(X) >= points:
                break

    X = np.array(X)
    y = np.array(y)

    # Train a polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    return model


def get_inc_angle_estimator(burst: burst.Burst, points=1000, min_height=0.0, max_height=2000.0, degree=2):

    pnts = get_random_points(points*2, burst.footprint, min_height, max_height)

    # Collect features (x, y, height) and targets (perpendicular baseline)
    X = []
    y = []
    for lat, lon, h in pnts:
        geocentric = coordinates.geodetic_to_geocentric(lat, lon, h)
        x_pixel, y_pixel = burst.pixel_from_geocentric(geocentric)
        if burst.is_valid(x_pixel, y_pixel):
            primary_pos = get_satellite_position(geocentric, burst)
            inc_angle = calculate_incidence_angle(geocentric, primary_pos)

            X.append([x_pixel, y_pixel, h])
            y.append(inc_angle)
            if len(X) >= points:
                break

    X = np.array(X)
    y = np.array(y)

    # Train a polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    return model

class Baseline:
    def __init__(self, primary_meta: metadata.MetaData, secondary_meta: metadata.MetaData, points=1000, min_height=0.0, max_height=2000.0, degree=3):
        self.__model = get_perpendicular_baseline_estimator(primary_meta.burst, secondary_meta.burst, points, min_height, max_height, degree)
        self.temporal_baseline = (secondary_meta.acquisition_date - primary_meta.acquisition_date).days

    def perpendicular_baseline(self, x: float, y: float, height: float = 0.0) -> float:
        return self.__model.predict([[x, y, height]])