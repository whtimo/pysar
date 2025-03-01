from pysar.sar import metadata, slc
from pysar import coordinates
from pyproj import Geod
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import map_coordinates
from rasterio.transform import from_origin
from rasterio import rpc
import json

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


def latlon_to_pixel(lat, lon, height, rpc):
    # Normalize ground coordinates
    X = (lat - rpc['offsets_scales']['X'][0]) / rpc['offsets_scales']['X'][1]
    Y = (lon - rpc['offsets_scales']['Y'][0]) / rpc['offsets_scales']['Y'][1]
    Z = (height - rpc['offsets_scales']['Z'][0]) / rpc['offsets_scales']['Z'][1]

    terms = generate_numerator_terms(X, Y, Z)

    # Calculate normalized row
    row_num = np.dot(rpc['row_num'], terms)
    row_den = np.dot(rpc['row_den'], terms)
    row_norm = row_num / row_den

    # Calculate normalized column
    col_num = np.dot(rpc['col_num'], terms)
    col_den = np.dot(rpc['col_den'], terms)
    col_norm = col_num / col_den

    # Denormalize pixel coordinates
    row = row_norm * rpc['offsets_scales']['row'][1] + rpc['offsets_scales']['row'][0]
    col = col_norm * rpc['offsets_scales']['col'][1] + rpc['offsets_scales']['col'][0]

    return row, col


# def pixel_to_latlon(row, col, height, rpc, max_iter=50, tol=1e-6):
#     # Normalize pixel coordinates
#     row_norm = (row - rpc['offsets_scales']['row'][0]) / rpc['offsets_scales']['row'][1]
#     col_norm = (col - rpc['offsets_scales']['col'][0]) / rpc['offsets_scales']['col'][1]
#     Z = (height - rpc['offsets_scales']['Z'][0]) / rpc['offsets_scales']['Z'][1]
#
#     # Initial guess (normalized coordinates)
#     X = 0.0
#     Y = 0.0
#
#     for _ in range(max_iter):
#         terms = generate_numerator_terms(X, Y, Z)
#
#         # Forward calculation
#         row_num = np.dot(rpc['row_num'], terms)
#         row_den = np.dot(rpc['row_den'], terms)
#         row_current = row_num / row_den
#
#         col_num = np.dot(rpc['col_num'], terms)
#         col_den = np.dot(rpc['col_den'], terms)
#         col_current = col_num / col_den
#
#         # Check convergence
#         if (abs(row_current - row_norm) < tol and
#                 abs(col_current - col_norm) < tol):
#             break
#
#         # Numerical Jacobian calculation
#         delta = 1e-8
#         terms_X = generate_numerator_terms(X + delta, Y, Z)
#         terms_Y = generate_numerator_terms(X, Y + delta, Z)
#
#         # Row partial derivatives
#         dRdX = (np.dot(rpc['row_num'], terms_X) / np.dot(rpc['row_den'], terms_X) - row_current
#                 dRdY = (np.dot(rpc['row_num'], terms_Y) / np.dot(rpc['row_den'], terms_Y) - row_current
#
#         # Column partial derivatives
#         dCdX = (np.dot(rpc['col_num'], terms_X) / np.dot(rpc['col_den'], terms_X) - col_current
#                 dCdY = (np.dot(rpc['col_num'], terms_Y) / np.dot(rpc['col_den'], terms_Y) - col_current
#
#         # Solve for update
#         J = np.array([[dRdX / delta, dRdY / delta],
#                       [dCdX / delta, dCdY / delta]])
#
#         residual = np.array([row_norm - row_current,
#                              col_norm - col_current])
#
#         try:
#             update = np.linalg.solve(J, residual)
#         except np.linalg.LinAlgError:
#             update = np.linalg.lstsq(J, residual, rcond=None)[0]
#
#         X += update[0]
#         Y += update[1]
#
#     # Denormalize coordinates
#     lon = X * rpc['offsets_scales']['X'][1] + rpc['offsets_scales']['X'][0]
#     lat = Y * rpc['offsets_scales']['Y'][1] + rpc['offsets_scales']['Y'][0]
#
#     return lat, lon

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
            'X': (offset_Y, scale_Y),
            'Y': (offset_X, scale_X),
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
    output_rpc = '/Users/timo/Documents/rapa_nui_tsx_rpc.rpc'

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
    rpc_c = rasterio.rpc.RPC(line_off=rpc['offsets_scales']['row'][0], samp_off=rpc['offsets_scales']['col'][0],
                     lat_off=rpc['offsets_scales']['X'][0], long_off=rpc['offsets_scales']['Y'][0],
                     height_off=rpc['offsets_scales']['Z'][0], line_scale=rpc['offsets_scales']['row'][1],
                     samp_scale=rpc['offsets_scales']['col'][1], lat_scale=rpc['offsets_scales']['X'][1],
                     long_scale=rpc['offsets_scales']['Y'][1], height_scale=rpc['offsets_scales']['Z'][1],
                     line_num_coeff=rpc['row_num'], line_den_coeff=rpc['row_den'],
                     samp_num_coeff=rpc['col_num'], samp_den_coeff=rpc['col_den'])


    pixx, pixy = latlon_to_pixel(slc.metadata.footprint.top(), slc.metadata.footprint.left(), 0.0, rpc)
    geoc2 = coordinates.geodetic_to_geocentric(slc.metadata.footprint.top(), slc.metadata.footprint.left(), 0.0)
    pixx2, pixy2 = slc.metadata.pixel_from_geocentric(geoc2)

    # Extract the coefficients and offsets/scales
    row_num = rpc['row_num']
    row_den = rpc['row_den']
    col_num = rpc['col_num']
    col_den = rpc['col_den']
    offsets_scales = rpc['offsets_scales']

    # Extract individual offsets and scales
    offset_X, scale_X = offsets_scales['X']
    offset_Y, scale_Y = offsets_scales['Y']
    offset_Z, scale_Z = offsets_scales['Z']
    offset_row, scale_row = offsets_scales['row']
    offset_col, scale_col = offsets_scales['col']

    # Generate the rpc_metadata dictionary
    rpc_metadata = {
        'LINE_OFF': offset_row,
        'SAMP_OFF': offset_col,
        'LAT_OFF': offset_Y,
        'LONG_OFF': offset_X,
        'HEIGHT_OFF': offset_Z,
        'LINE_SCALE': scale_row,
        'SAMP_SCALE': scale_col,
        'LAT_SCALE': scale_Y,
        'LONG_SCALE': scale_X,
        'HEIGHT_SCALE': scale_Z,
        # 'LINE_NUM_COEFF': ' '.join(map(str, row_num)),
        # 'LINE_DEN_COEFF': ' '.join(map(str, row_den)),
        # 'SAMP_NUM_COEFF': ' '.join(map(str, col_num)),
        # 'SAMP_DEN_COEFF': ' '.join(map(str, col_den))
        'LINE_NUM_COEFF': row_num,
        'LINE_DEN_COEFF': row_den,
        'SAMP_NUM_COEFF': col_num,
        'SAMP_DEN_COEFF': col_den
    }

    rpc_json = json.dumps(rpc)

    meta = {
        'driver': 'GTiff',  # GeoTIFF driver
        'dtype': sar_data.dtype,  # Data type of the array
        'nodata': None,  # No data value
        'width': sar_data.shape[1],  # Width of the raster
        'height': sar_data.shape[0],  # Height of the raster
        'count': 1,  # Number of bands
        'crs': 'EPSG:4326',  # Coordinate Reference System
        'transform': from_origin(0, sar_data.shape[0], 1, 1)


    }

    # Save the raster file with RPC metadata
    with rasterio.open(output_geo_tiff, 'w', **meta) as dst:
        dst.write(sar_data, 1)  # Write the data to the first band
        dst.update_tags(ns='RPC', RPC00B=rpc_json)

        # Add RPC metadata to the file's tags
        #dst.update_tags(**rpc_metadata)
        #dst.rpc = rpc_metadata

    # Save the image with RPC metadata
    # with rasterio.open(
    #         output_geo_tiff,
    #         'w',
    #         driver='GTiff',
    #         height=sar_data.shape[0],
    #         width=sar_data.shape[1],
    #         count=1,  # Number of bands
    #         dtype=sar_data.dtype
    #         #crs='+proj=latlong',  # Coordinate reference system (adjust as needed)
    #         #transform=transform,
    #         #rpc = rpc_c
    #
    # ) as dst:
    #
    #
    #     # # Add RPC metadata
    #     # dst.rpc = {
    #     #     'LINE_OFF': rpc['offsets_scales']['row'][0],
    #     #     'SAMP_OFF': rpc['offsets_scales']['col'][0],
    #     #     'LAT_OFF': rpc['offsets_scales']['X'][0],
    #     #     'LONG_OFF': rpc['offsets_scales']['Y'][0],
    #     #     'HEIGHT_OFF': rpc['offsets_scales']['Z'][0],
    #     #     'LINE_SCALE': rpc['offsets_scales']['row'][1],
    #     #     'SAMP_SCALE': rpc['offsets_scales']['col'][1],
    #     #     'LAT_SCALE': rpc['offsets_scales']['X'][1],
    #     #     'LONG_SCALE': rpc['offsets_scales']['Y'][1],
    #     #     'HEIGHT_SCALE': rpc['offsets_scales']['Z'][1],
    #     #     'LINE_NUM_COEFF': rpc['row_num'],
    #     #     'LINE_DEN_COEFF': rpc['row_den'],
    #     #     'SAMP_NUM_COEFF': rpc['col_num'],
    #     #     'SAMP_DEN_COEFF': rpc['col_den']
    #     # }
    #
    #     # Write the image data
    #     dst.write(sar_data, 1)
    #
    # with open(output_rpc, 'w') as f:
    #     # Write offsets and scales
    #     f.write(f"LINE_OFF: {rpc['offsets_scales']['row'][0]:+012.2f} pixels\n")
    #     f.write(f"SAMP_OFF: {rpc['offsets_scales']['col'][0]:+012.2f} pixels\n")
    #     f.write(f"LAT_OFF: {rpc['offsets_scales']['X'][0]:+012.8f} degrees\n")
    #     f.write(f"LONG_OFF: {rpc['offsets_scales']['Y'][0]:+012.8f} degrees\n")
    #     f.write(f"HEIGHT_OFF: {rpc['offsets_scales']['Z'][0]:+012.3f} meters\n")
    #     f.write(f"LINE_SCALE: {rpc['offsets_scales']['row'][1]:+012.2f} pixels\n")
    #     f.write(f"SAMP_SCALE: {rpc['offsets_scales']['col'][1]:+012.2f} pixels\n")
    #     f.write(f"LAT_SCALE: {rpc['offsets_scales']['X'][1]:+012.8f} degrees\n")
    #     f.write(f"LONG_SCALE: {rpc['offsets_scales']['Y'][1]:+012.8f} degrees\n")
    #     f.write(f"HEIGHT_SCALE: {rpc['offsets_scales']['Z'][1]:+012.3f} meters\n")
    #
    #     # Write line numerator coefficients
    #     for i, coeff in enumerate(rpc['row_num'], start=1):
    #         f.write(f"LINE_NUM_COEFF_{i}: {coeff:+.16E}\n")
    #
    #     # Write line denominator coefficients
    #     for i, coeff in enumerate(rpc['row_den'], start=1):
    #         f.write(f"LINE_DEN_COEFF_{i}: {coeff:+.16E}\n")
    #
    #     # Write sample numerator coefficients
    #     for i, coeff in enumerate(rpc['col_num'], start=1):
    #         f.write(f"SAMP_NUM_COEFF_{i}: {coeff:+.16E}\n")
    #
    #     # Write sample denominator coefficients
    #     for i, coeff in enumerate(rpc['col_den'], start=1):
    #         f.write(f"SAMP_DEN_COEFF_{i}: {coeff:+.16E}\n")

    # with open(output_rpc, 'w') as f:
    #     # Write header
    #     f.write(f'satId = "TSX";\n')
    #     f.write(f'bandId = "SAR";\n')
    #     f.write(f'SpecId = "RPC";\n')
    #     f.write("BEGIN_GROUP = IMAGE\n")
    #
    #     # Write error values
    #     f.write(f"\terrBias = 1.0;\n")
    #     f.write(f"\terrRand = 0.0;\n")
    #
    #     # Write offsets and scales
    #     f.write(f"\tlineOffset = {rpc['offsets_scales']['row'][0]:.6f};\n")
    #     f.write(f"\tsampOffset = {rpc['offsets_scales']['col'][0]:.6f};\n")
    #     f.write(f"\tlatOffset = {rpc['offsets_scales']['X'][0]:.8f};\n")
    #     f.write(f"\tlongOffset = {rpc['offsets_scales']['Y'][0]:.8f};\n")
    #     f.write(f"\theightOffset = {rpc['offsets_scales']['Z'][0]:.8f};\n")
    #     f.write(f"\tlineScale = {rpc['offsets_scales']['row'][1]:.6f};\n")
    #     f.write(f"\tsampScale = {rpc['offsets_scales']['col'][1]:.6f};\n")
    #     f.write(f"\tlatScale = {rpc['offsets_scales']['X'][1]:.8f};\n")
    #     f.write(f"\tlongScale = {rpc['offsets_scales']['Y'][1]:.8f};\n")
    #     f.write(f"\theightScale = {rpc['offsets_scales']['Z'][1]:.6f};\n")
    #
    #     # Write line numerator coefficients
    #     f.write("\tlineNumCoef = (\n")
    #     for i, coeff in enumerate(rpc['row_num']):
    #         f.write(f"\t\t{coeff:.11E},\n" if i < len(rpc['row_num']) - 1 else f"\t\t{coeff:.11E});\n")
    #
    #     # Write line denominator coefficients
    #     f.write("\tlineDenCoef = (\n")
    #     for i, coeff in enumerate(rpc['row_den']):
    #         f.write(f"\t\t{coeff:.11E},\n" if i < len(rpc['row_den']) - 1 else f"\t\t{coeff:.11E});\n")
    #
    #     # Write sample numerator coefficients
    #     f.write("\tsampNumCoef = (\n")
    #     for i, coeff in enumerate(rpc['col_num']):
    #         f.write(f"\t\t{coeff:.11E},\n" if i < len(rpc['col_num']) - 1 else f"\t\t{coeff:.11E});\n")
    #
    #     # Write sample denominator coefficients
    #     f.write("\tsampDenCoef = (\n")
    #     for i, coeff in enumerate(rpc['col_den']):
    #         f.write(f"\t\t{coeff:.11E},\n" if i < len(rpc['col_den']) - 1 else f"\t\t{coeff:.11E});\n")
    #
    #     # Write footer
    #     f.write("END_GROUP = IMAGE\n")
    #     f.write("END;\n")