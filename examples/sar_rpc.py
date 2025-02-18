from pysar.sar import metadata, slc
from pysar import coordinates
from pyproj import Geod
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import map_coordinates
from rasterio.transform import from_origin

def generate_numerator_terms(X, Y, Z):
    return [
        1,
        X, Y, Z,
        X * Y, X * Z, Y * Z,
        X ** 2, Y ** 2, Z ** 2,
        X ** 3, X ** 2 * Y, X ** 2 * Z,
        X * Y ** 2, X * Y * Z, X * Z ** 2,
        Y ** 3, Y ** 2 * Z, Y * Z ** 2, Z ** 3
    ]


def calculate_rpc(ground_coords, pixel_coords):
    ground_coords = np.array(ground_coords)
    pixel_coords = np.array(pixel_coords)

    # Normalize ground coordinates
    min_X, max_X = np.min(ground_coords[:, 0]), np.max(ground_coords[:, 0])
    min_Y, max_Y = np.min(ground_coords[:, 1]), np.max(ground_coords[:, 1])
    min_Z, max_Z = np.min(ground_coords[:, 2]), np.max(ground_coords[:, 2])

    offset_X, scale_X = (min_X + max_X) / 2, (max_X - min_X) / 2
    offset_Y, scale_Y = (min_Y + max_Y) / 2, (max_Y - min_Y) / 2
    offset_Z, scale_Z = (min_Z + max_Z) / 2, (max_Z - min_Z) / 2

    X_norm = (ground_coords[:, 0] - offset_X) / scale_X
    Y_norm = (ground_coords[:, 1] - offset_Y) / scale_Y
    Z_norm = (ground_coords[:, 2] - offset_Z) / scale_Z

    # Normalize pixel coordinates
    min_row, max_row = np.min(pixel_coords[:, 0]), np.max(pixel_coords[:, 0])
    min_col, max_col = np.min(pixel_coords[:, 1]), np.max(pixel_coords[:, 1])

    offset_row, scale_row = (min_row + max_row) / 2, (max_row - min_row) / 2
    offset_col, scale_col = (min_col + max_col) / 2, (max_col - min_col) / 2

    row_norm = (pixel_coords[:, 0] - offset_row) / scale_row
    col_norm = (pixel_coords[:, 1] - offset_col) / scale_col

    # Build matrices for row and column equations
    num_samples = len(ground_coords)
    A_row = np.zeros((num_samples, 39))
    A_col = np.zeros((num_samples, 39))

    for i in range(num_samples):
        X, Y, Z = X_norm[i], Y_norm[i], Z_norm[i]
        r, c = row_norm[i], col_norm[i]

        # Numerator terms (20)
        num_terms = generate_numerator_terms(X, Y, Z)

        # Denominator terms (19, exclude constant)
        den_terms = num_terms[1:]

        # Row equation
        A_row[i, :20] = num_terms
        A_row[i, 20:] = [-r * t for t in den_terms]

        # Column equation
        A_col[i, :20] = num_terms
        A_col[i, 20:] = [-c * t for t in den_terms]

    # Solve least squares
    coeff_row, _, _, _ = np.linalg.lstsq(A_row, row_norm, rcond=None)
    coeff_col, _, _, _ = np.linalg.lstsq(A_col, col_norm, rcond=None)

    # Extract coefficients
    rpc = {
        'row_num': coeff_row[:20].tolist(),
        'row_den': [1.0] + coeff_row[20:].tolist(),  # Include denominator's 1
        'col_num': coeff_col[:20].tolist(),
        'col_den': [1.0] + coeff_col[20:].tolist(),
        'offsets_scales': {
            'X': (offset_X, scale_X),
            'Y': (offset_Y, scale_Y),
            'Z': (offset_Z, scale_Z),
            'row': (offset_row, scale_row),
            'col': (offset_col, scale_col)
        }
    }

    return rpc

if __name__ == "__main__":

    tsx_path = '/Users/timo/Documents/Rapa Nui/dims_op_oc_dfd2_693810856_1/TSX-1.SAR.L1B/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450.xml'
    #srtm_path = '/Users/timo/Documents/srtm_15_18/srtm_15_18.tif'

    output_geo_tiff = '/Users/timo/Documents/rapa_nui_tsx_rpc.tiff'

    slc_full = slc.fromTSX(tsx_path, 0)
    slc = slc_full.multilook(1, 4)


    sar_buffer = slc.slcdata.read()
    sar_data = np.abs(sar_buffer)

    print('Generate array')
    lons = np.linspace(slc.metadata.footprint.left(), slc.metadata.footprint.right(), 400)
    lats = np.linspace(slc.metadata.footprint.top(), slc.metadata.footprint.bottom(), 400)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    height_grid = np.random.uniform(low=0, high=2000, size=(400, 400))

    print('Transform into geocentric coordinates')
    geoc = coordinates.geodetic_to_geocentric(lat_grid, lon_grid, height_grid)

    print('Calculate SAR image coordinates')
    sar_x, sar_y = slc.metadata.pixel_from_geocentric(geoc)

    ground = []
    pix = []

    for i in range(40):
        for j in range(40):
            ground.append((lat_grid[i, j], lon_grid[i, j], height_grid[i, j]))
            pix.append((sar_x[i, j], sar_y[i, j]))

    rpc = calculate_rpc(ground, pix)

    transform = from_origin(0, 0, 1, 1)  # Origin at (0, 0), pixel size 1x1

    # Save the image with RPC metadata
    with rasterio.open(
            'output_with_rpc.tif',
            'w',
            driver='GTiff',
            height=sar_data.shape[1],
            width=sar_data.shape[0],
            count=1,  # Number of bands
            dtype=sar_data.dtype,
            crs='+proj=latlong',  # Coordinate reference system (adjust as needed)
            transform=transform
    ) as dst:
        # Write the image data
        dst.write(sar_data, 1)

        # Add RPC metadata
        dst.rpc = {
            'LINE_OFF': rpc['offsets_scales']['row'][0],
            'SAMP_OFF': rpc['offsets_scales']['col'][0],
            'LAT_OFF': rpc['offsets_scales']['Y'][0],
            'LONG_OFF': rpc['offsets_scales']['X'][0],
            'HEIGHT_OFF': rpc['offsets_scales']['Z'][0],
            'LINE_SCALE': rpc['offsets_scales']['row'][1],
            'SAMP_SCALE': rpc['offsets_scales']['col'][1],
            'LAT_SCALE': rpc['offsets_scales']['Y'][1],
            'LONG_SCALE': rpc['offsets_scales']['X'][1],
            'HEIGHT_SCALE': rpc['offsets_scales']['Z'][1],
            'LINE_NUM_COEFF': rpc['row_num'],
            'LINE_DEN_COEFF': rpc['row_den'],
            'SAMP_NUM_COEFF': rpc['col_num'],
            'SAMP_DEN_COEFF': rpc['col_den']
        }

