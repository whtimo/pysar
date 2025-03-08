from pysar.sar import slc
from pysar import coordinates
from pyproj import Geod
import numpy as np
import rasterio
from scipy.ndimage import map_coordinates
from rasterio.transform import from_origin

if __name__ == "__main__":

    tsx_path = ''
    srtm_path = ''

    output_geo_tiff = ''

    slc_full = slc.fromTSX(tsx_path, 0)
    slc = slc_full.multilook(1, 4)

    geod = Geod(ellps="WGS84")

    _, _, distance_range = geod.inv(slc.metadata.footprint.upper_left[1], slc.metadata.footprint.upper_left[0],
                                          slc.metadata.footprint.upper_right[1], slc.metadata.footprint.upper_right[0])

    _, _, distance_az = geod.inv(slc.metadata.footprint.upper_left[1], slc.metadata.footprint.upper_left[0],
                                          slc.metadata.footprint.lower_left[1], slc.metadata.footprint.lower_left[0])

    pxi_size_range = distance_range / slc.metadata.number_columns
    pxi_size_az = distance_az / slc.metadata.number_rows
    pix_size = np.round(max(pxi_size_range, pxi_size_az), 1)
    print(f'Georeferencing to image with pixel size of {pix_size}')

    _, _, distance_width = geod.inv(slc.metadata.footprint.left(), slc.metadata.footprint.upper_left[0],
                                    slc.metadata.footprint.right(), slc.metadata.footprint.upper_left[0])

    _, _, distance_height = geod.inv(slc.metadata.footprint.upper_left[1], slc.metadata.footprint.top(),
                                    slc.metadata.footprint.upper_left[1], slc.metadata.footprint.bottom())

    new_width = int(np.round(distance_width / pix_size))
    new_height = int(np.round(distance_height / pix_size))

    utm_zone = coordinates.get_utm_zone(slc.metadata.footprint.left())
    transformer_geo = coordinates.get_geodetic_to_utm_transformer(utm_zone)
    transformer = coordinates.get_utm_to_geodetic_transformer(utm_zone)
    ll_utm = transformer_geo.transform(slc.metadata.footprint.left(), slc.metadata.footprint.bottom())
    ul_utm = transformer_geo.transform(slc.metadata.footprint.left(), slc.metadata.footprint.top())

    print('Read data')
    with rasterio.open(srtm_path) as srtm:
        # Convert the transformed coordinates to row/column indices in the raster
        srtm_transform = srtm.transform
        srtm_data = np.array(srtm.read(1), dtype=np.float32)

    sar_buffer = slc.slcdata.read()
    sar_data = np.abs(sar_buffer)

    print('Generate array')
    east = np.linspace(ll_utm[0], ll_utm[0] + pix_size * (new_width - 1), new_width)
    north = np.linspace(ll_utm[1], ll_utm[1] + pix_size * (new_height - 1), new_height)
    east_grid, north_grid = np.meshgrid(east, north)

    print('Get heights')
    geo_grid_lon, geo_grid_lat = transformer.transform(east_grid, north_grid)
    col_grid, row_grid = ~srtm_transform * (geo_grid_lon, geo_grid_lat)
    height_grid = map_coordinates(srtm_data, [[row_grid], [col_grid]], order=1, mode='nearest')[0]

    print('Transform into geocentric coordinates')
    geoc = coordinates.geodetic_to_geocentric(geo_grid_lat, geo_grid_lon, height_grid)

    print('Calculate SAR image coordinates')
    sar_x, sar_y = slc.metadata.pixel_from_geocentric(geoc)

    print('Interpolate SAR image')
    new_geo_sar = map_coordinates(sar_data, [[sar_y], [sar_x]], order=2, mode='constant', cval=0)[0]
    new_geo_sar = new_geo_sar[::-1, :]

    print('Write image')
    utm_crs = f"EPSG:326{utm_zone}"  # EPSG:326XX for northern hemisphere, EPSG:327XX for southern

    # Define the transform (pixel coordinates to geographic coordinates)
    # Example: 100x100 image with 10m resolution, starting at (easting=500000, northing=4500000)
    transform = from_origin(ul_utm[0], ul_utm[1], pix_size, pix_size)  # (x_min, y_max, x_res, y_res)

    # Save the array as a GeoTIFF
    with rasterio.open(
        output_geo_tiff,  # Output file path
        "w",  # Write mode
        driver="GTiff",  # GeoTIFF format
        height=new_height,  # Number of rows
        width=new_width,  # Number of columns
        count=1,  # Number of bands
        dtype=np.float32,  # Data type (float32)
        crs=utm_crs,  # Coordinate reference system
        transform=transform,  # Affine transform
    ) as dst:
        dst.write(new_geo_sar, 1)

    print('end')# Write the data to the first band