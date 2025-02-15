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

def get_utm_zone(lon):
    utm_zone = (int((lon + 180) / 6) + 1)
    return utm_zone

def get_geodetic_to_utm_transformer(utm_zone):
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Create a transformer to convert from geographic (lat/lon) to UTM
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    return transformer

def get_utm_to_geodetic_transformer(utm_zone):
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Create a transformer to convert from geographic (lat/lon) to UTM
    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    return transformer

def geodetic_to_utm(lat, lon, utm_zone=None):
    if utm_zone is None:
        utm_zone = get_utm_zone(lon)

    transformer = get_geodetic_to_utm_transformer(utm_zone)
    # Transform the geographic coordinate to UTM
    easting, northing = transformer.transform(lon, lat)
    return easting, northing

def utm_to_geodetic(easting, norting, utm_zone: int):
    transformer = get_utm_to_geodetic_transformer(utm_zone)
    # Transform the geographic coordinate to UTM
    lat, lon = transformer.transform(easting, northing)
    return lat, lon