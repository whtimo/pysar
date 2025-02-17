from pysar.sar import metadata, slc
from pysar.insar import resampled_pair
from pysar import coordinates
from pyproj import Geod
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import map_coordinates
from rasterio.transform import from_origin

if __name__ == "__main__":

    file_pair = '/Users/timo/Documents/WuhanEast/pysar/TDX-1_0_2018-07-24__TDX-1_2018-08-26.pysar.resampled.xml'
    srtm_path = '/Users/timo/Documents/srtm_15_18/srtm_15_18.tif'
    output_path = '/Users/timo/Documents/WuhanEast/pysar'

    tsx_path = '/Users/timo/Documents/Rapa Nui/dims_op_oc_dfd2_693810856_1/TSX-1.SAR.L1B/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450/TDX1_SAR__SSC______ST_S_SRA_20231005T014450_20231005T014450.xml'

    slc = slc.fromTSX(tsx_path, 0)
    #pair = resampled_pair.ResampledPair(file_pair)

    with rasterio.open(srtm_path) as srtm:
        # Convert the transformed coordinates to row/column indices in the raster
        srtm_transform = srtm.transform
        srtm_data = np.array(srtm.read(1), dtype=np.float32)

    tl_x, tl_y = ~srtm_transform * (slc.metadata.footprint.left(), slc.metadata.footprint.top())
    br_x, br_y = ~srtm_transform * (slc.metadata.footprint.right(), slc.metadata.footprint.bottom())
    min_x = int(np.min([tl_x, br_x])) - 1
    min_y = int(np.min([tl_y, br_y])) - 1
    max_x = int(np.max([tl_x, br_x])) + 1
    max_y = int(np.max([tl_y, br_y])) + 1
    if min_x < 0: min_x = 0
    if min_y < 0: min_y = 0
    if max_x >= srtm_data.shape[1]: max_x = srtm_data.shape[1] - 1
    if max_y >= srtm_data.shape[0]: max_y = srtm_data.shape[0] - 1
    range_x = max_x - min_x
    range_y = max_y - min_y

    x = np.linspace(min_x, max_x, range_x, dtype=np.int32)
    y = np.linspace(min_y, max_y, range_y, dtype=np.int32)
    x_grid, y_grid = np.meshgrid(x, y)
    lons, lats = srtm_transform * (x_grid, y_grid)
    heights = map_coordinates(srtm_data, [[y_grid], [x_grid]], order=1, mode='nearest')[0]

    print('Transform into geocentric coordinates')
    geoc = coordinates.geodetic_to_geocentric(lats, lons, heights)

    print('Calculate SAR image coordinates')
    sar_x, sar_y = slc.metadata.pixel_from_geocentric(geoc)

    print('')



    print('end')