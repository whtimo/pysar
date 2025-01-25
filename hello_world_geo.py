#Required Libraries
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.vrt import WarpedVRT

def normalize(arr):
    min_vals = arr.min()
    max_vals = arr.max()
    ranges = max_vals - min_vals
    normalized_arr = (arr - min_vals) / ranges
    return normalized_arr

# Load the image using PIL
filename = '/Users/timo/Documents/data_radar_book/South Africa/1012402560010001_R1_01/SVN1-02_20241206_L1B0001320179_1012402560010001_R1_01/SVN1-02_20241206_L1B0001320179_1012402560010001_R1_01-MUX1.tiff'
with rasterio.open(filename) as img:
    # Create a WarpeVRT to reproject using RPCs
    with WarpedVRT(img, crs='EPSG:4326', rpc=True) as vrt:
        # Read the first band (adjust if using multiple bands)
        data = vrt.read([3, 2, 1])
        data = np.moveaxis(data, 0, 2)

        print(f'Image shape: {data.shape}')

        scaled_data = np.zeros(data.shape, dtype=np.float32)
        scaled_data[:, :, 0] = normalize(data[:, :, 0])
        scaled_data[:, :, 1] = normalize(data[:, :, 1])
        scaled_data[:, :, 2] = normalize(data[:, :, 2])

        # Calculate geographic bounds
        left, bottom, right, top = rasterio.transform.array_bounds(
            vrt.height, vrt.width, vrt.transform
        )

        # Plot the image with geographic coordinates
        plt.figure(figsize=(12, 8))
        img = plt.imshow(scaled_data, extent=(left, right, bottom, top))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
