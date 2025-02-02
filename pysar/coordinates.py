from pyproj import Transformer


def geodetic_to_geocentric(lat, lon, height=0):
    """
    Convert geodetic coordinates (latitude, longitude, height) to geocentric coordinates (ECEF).

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        height (float): Height above the ellipsoid in meters (default: 0).

    Returns:
        tuple: Geocentric coordinates (x, y, z) in meters.
    """
    # Define a transformer for geodetic to ECEF conversion
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    # Perform the transformation
    x, y, z = transformer.transform(lon, lat, height)
    return x, y, z
