from pysar.sar import slc
from pysar import coordinates
import numpy as np

import xml.etree.ElementTree as ET


def quadratic_subpixel_peak(window):
    # Find the integer peak
    i_peak, j_peak = np.unravel_index(np.argmax(window), window.shape)

    # Check if 3x3 can be extracted around the peak
    if (i_peak < 1 or i_peak >= window.shape[0] - 1 or
            j_peak < 1 or j_peak >= window.shape[1] - 1):
        return (i_peak + 0.0, j_peak + 0.0)  # Return as float

    # Extract 3x3 region around the peak
    window_3x3 = window[i_peak - 1:i_peak + 2, j_peak - 1:j_peak + 2]

    # Coordinates relative to the center (peak at 1,1 in 3x3)
    y, x = np.indices((3, 3))
    x = x - 1  # [-1, 0, 1]
    y = y - 1  # [-1, 0, 1]

    # Flatten and build the design matrix
    X = x.flatten()
    Y = y.flatten()
    Z = window_3x3.flatten()
    M = np.vstack([X ** 2, Y ** 2, X * Y, X, Y, np.ones(len(Z))]).T

    # Solve least squares
    try:
        coeff = np.linalg.lstsq(M, Z, rcond=None)[0]
    except np.linalg.LinAlgError:
        return (i_peak + 0.0, j_peak + 0.0)

    a, b, c, d, e, _ = coeff
    denominator = 4 * a * b - c ** 2
    if denominator == 0:
        return (i_peak + 0.0, j_peak + 0.0)

    # Compute sub-pixel offsets
    delta_x = (-2 * b * d + c * e) / denominator
    delta_y = (-2 * a * e + c * d) / denominator

    # Clip deltas to stay within ±1 pixel
    delta_x = np.clip(delta_x, -1, 1)
    delta_y = np.clip(delta_y, -1, 1)

    return (i_peak + delta_y, j_peak + delta_x)


def fft_subpixel_peak(window, upsample_factor=4):
    # Find integer peak
    i_peak, j_peak = np.unravel_index(np.argmax(window), window.shape)

    # Extract a 3x3 region around the peak
    if (i_peak < 1 or i_peak >= window.shape[0] - 1 or
            j_peak < 1 or j_peak >= window.shape[1] - 1):
        return (i_peak + 0.0, j_peak + 0.0)
    small_window = window[i_peak - 1:i_peak + 2, j_peak - 1:j_peak + 2]

    # Upsample using FFT
    n_rows, n_cols = small_window.shape
    window_fft = np.fft.fft2(small_window)

    # Pad the FFT with zeros
    padded_size = (n_rows * upsample_factor, n_cols * upsample_factor)
    padded_fft = np.zeros(padded_size, dtype=complex)
    padded_fft[:n_rows, :n_cols] = window_fft  # Place the low frequencies

    # Inverse FFT to get upsampled image
    upsampled = np.fft.ifft2(padded_fft)
    upsampled_mag = np.abs(upsampled)

    # Find peak in upsampled image
    i_peak_up, j_peak_up = np.unravel_index(np.argmax(upsampled_mag), upsampled_mag.shape)

    # Convert to sub-pixel offset in the original window
    delta_i = (i_peak_up / upsample_factor) - 1  # Offset from 3x3 center
    delta_j = (j_peak_up / upsample_factor) - 1
    return (i_peak + delta_i, j_peak + delta_j)

if __name__ == "__main__":

    file_path = '/home/timo/Projects/ReceivingStation/dims_op_oc_dfd2_693818609_1/TSX-1.SAR.L1B/TDX1_SAR__SSC______ST_S_SRA_20230926T223006_20230926T223007/TDX1_SAR__SSC______ST_S_SRA_20230926T223006_20230926T223007.xml'
    coord_path = '/home/timo/Projects/ReceivingStation/SRS/222222.kml'


    slc = slc.fromTSX(file_path, 0)

    root = ET.parse(coord_path)
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Extract all Placemark elements
    placemarks = root.findall('.//kml:Placemark', namespace)

    # List to store the extracted place information
    places = []

    # Iterate through each Placemark and extract the required information
    for placemark in placemarks:
        name = placemark.find('kml:name', namespace).text
        coord = placemark.find('.//kml:coordinates', namespace).text.strip()
        lon, lat, height = map(float, coord.split(','))

        places.append({
            'name': name,
            'lat': float(lat),
            'lon': float(lon),
            'height': float(height)
        })

    print('Transform into geocentric coordinates')
    geoc = coordinates.geodetic_to_geocentric(places[1]['lat'], places[1]['lon'], places[1]['height'])

    sar_x, sar_y = slc.metadata.pixel_from_geocentric(geoc)
    print(f'Estimated coordinates: {sar_x} / {sar_y}')
    print('Coordinates from bzar2: 3326.57 / 8579.54')
    print(f'Difference: {sar_x - 3326.57} / {sar_y - 8579.54}')

    data = np.abs(slc.slcdata.read())

    window_size = 25  # Must be odd; adjust based on expected feature size
    x, y = int(sar_x), int(sar_y)  # Example coordinate

    # Extract window around the coordinate
    window = data[y - window_size // 2: y + window_size // 2 + 1,
             x - window_size // 2: x + window_size // 2 + 1]

    # Compute sub-pixel peak using FFT upsampling
    subpixel_y_fft, subpixel_x_fft = fft_subpixel_peak(window, upsample_factor=64)
    global_x_fft = x - window_size // 2 + subpixel_x_fft
    global_y_fft = y - window_size // 2 + subpixel_y_fft

    print(f"FFT Upsampling: ({global_x_fft}, {global_y_fft})")

    print(f'Difference to data: {sar_x - global_x_fft} / {sar_y - global_y_fft}')
    print(f'Bzar Difference to data: {3326.57 - global_x_fft} / {8579.54 - global_y_fft}')

