#Required Libraries
import rasterio
import matplotlib.pyplot as plt
import numpy as np

def normalize(arr):
    min_vals = arr.min()
    max_vals = arr.max()
    ranges = max_vals - min_vals
    normalized_arr = (arr - min_vals) / ranges
    return normalized_arr

# Load the image using PIL
filename = '/Users/timo/Documents/data_radar_book/South Africa/1012402560010001_R1_01/SVN1-02_20241206_L1B0001320179_1012402560010001_R1_01/SVN1-02_20241206_L1B0001320179_1012402560010001_R1_01-MUX1.tiff'
with rasterio.open(filename) as img:
    data = img.read([3, 2, 1])
    data = np.moveaxis(data, 0, 2)
    print(f'Image shape: {data.shape}')

    scaled_data = np.zeros(data.shape, dtype=np.float32)
    scaled_data[:, :, 0] = normalize(data[:, :, 0])
    scaled_data[:, :, 1] = normalize(data[:, :, 1])
    scaled_data[:, :, 2] = normalize(data[:, :, 2])

    plt.figure(figsize=(6, 8))
    plt.imshow(scaled_data)
    plt.show()