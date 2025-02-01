#Required Libraries
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.vrt import WarpedVRT


# Load the image using PIL
filename = '/Users/timo/Documents/data_radar_book/Beijing SAR/1/SV2-03_MONO_KSBJ71_STRIP_006045_E116.7_N40.1_20230816_SLC_HH_L1A_0000364742/SV2-03_MONO_KSBJ71_STRIP_006045_E116.7_N40.1_20230816_SLC_HH_L1A_0000364742.tiff'
with rasterio.open(filename) as img:
    # Create a WarpeVRT to reproject using RPCs
    with WarpedVRT(img, crs='EPSG:4326', rpc=True) as vrt:
        # Read the first band (adjust if using multiple bands)
        data_i = vrt.read(1)
        data_q = vrt.read(2)
        data = np.empty(data_i.shape, dtype=np.complex64)
        data.real = data_i
        data.imag = data_q
        print(f'Image shape: {data.shape}')

        left, bottom, right, top = rasterio.transform.array_bounds(
            vrt.height, vrt.width, vrt.transform
        )

        # Plot the image with geographic coordinates
        plt.figure(figsize=(8, 6))
        img = plt.imshow(np.abs(data), extent=(left, right, bottom, top), cmap='gray', vmin=0, vmax=128)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig('/Users/timo/Documents/sar_geo.png', dpi=300, bbox_inches='tight')
        #plt.show()
