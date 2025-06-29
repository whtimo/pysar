import numpy as np
from scipy import fftpack
from pysar import coordinates, footprint
from pysar.sar import metadata
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.pipeline import make_pipeline
import pandas as pd

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt

def get_random_points(count: int, footprint: footprint.Footprint, height: float = 0):

    result = []
    for _ in range(count):
        lon = random.uniform(footprint.left(), footprint.right())
        lat = random.uniform(footprint.top(), footprint.bottom())
        result.append((lat, lon, height))

    return result

def orbit_shift(master: metadata.MetaData, slave: metadata.MetaData, height: float = 0.0):
    cent_lon = (master.footprint.left() + master.footprint.right()) / 2
    cent_lat = (master.footprint.top() + master.footprint.bottom()) / 2
    geocentric = coordinates.geodetic_to_geocentric(cent_lat, cent_lon, height)
    m_x, m_y = master.pixel_from_geocentric(geocentric)
    s_x, s_y = slave.pixel_from_geocentric(geocentric)

    return s_x - m_x, s_y - m_y

def subpixel_shift(master: np.ndarray, slave: np.ndarray, search_pix: int, search_line: int,  upsample_factor=16):
    """
    Compute sub-pixel shift between master and slave images.
    Args:
        master (ndarray): Smaller master window (2D array).
        slave (ndarray): Larger search window (2D array).
        upsample_factor (int): Factor for upsampling (default: 16).
    Returns:
        ndarray: Sub-pixel shift [dy, dx].
    """

    m, n = slave.shape
    upsampled_s = fftpack.fftshift(fftpack.fft2(slave, (m * upsample_factor, n * upsample_factor)))
    upsampled_slave = fftpack.ifft2(upsampled_s).real
    f_slave = fftpack.fft2(upsampled_slave, upsampled_slave.shape)

    max_corr = 0
    max_shifts = (0,0)

    for l in range(-search_line, search_line+1, 1):
        for p in range(-search_pix, search_pix+1, 1):
            # Upsample region using FFT
            y_sub = l + search_line
            x_sub = p + search_line

            upsampled_m = fftpack.fftshift(fftpack.fft2(master[l:l+m, p:p+n], (m*upsample_factor, n*upsample_factor)))
            upsampled_master = fftpack.ifft2(upsampled_m).real

            f_master = fftpack.fft2(upsampled_master, upsampled_master.shape)
            cross_corr = fftpack.fftshift(fftpack.ifft2(f_slave * np.conj(f_master))).real

            # Integer-pixel shift
            max_idx = np.argmax(cross_corr)
            peak = np.unravel_index(max_idx, cross_corr.shape)
            center = np.array(cross_corr.shape) // 2
            shifts = (np.array(peak) - center) / upsample_factor
            max_corr_sub = cross_corr[peak]
            if max_corr_sub > max_corr:
                max_corr = max_corr_sub
                max_shifts = (shifts[0] + p, shifts[1] + l)

    return max_shifts, max_corr

def subpixel_shifts(master_meta: metadata.MetaData, slave_meta: metadata.MetaData, coarse_shift_x: int, coarse_shift_y: int, master_data: np.ndarray, slave_data: np.ndarray, window_size_x:int = 128, window_size_y:int = 128, average_height: float = 0, points=1600, output = None) -> pd.DataFrame:

    pnts = get_random_points(points*2, master_meta.footprint, average_height)
    win_radius_x = (window_size_x - 1) // 2
    win_radius_y = (window_size_y - 1) // 2

    result = pd.DataFrame(columns=['masterX', 'masterY', 'shiftX', 'shiftY'])

    for lat, lon, h in pnts:
        geocentric = coordinates.geodetic_to_geocentric(lat, lon, h)
        m_x, m_y = master_meta.pixel_from_geocentric(geocentric)
        m_x, m_y = int(m_x), int(m_y)
        if master_meta.is_valid(m_x, m_y, window_size_x, window_size_y):
            s_x, s_y = slave_meta.pixel_from_geocentric(geocentric)
            s_x, s_y = int(s_x), int(s_y)
            if slave_meta.is_valid(s_x, s_y, window_size_x, window_size_y):
                mas_subset = np.abs(master_data[m_y - win_radius_y:m_y + win_radius_y, m_x - win_radius_x:m_x + win_radius_x])
                sl_subset = np.abs(slave_data[s_y - win_radius_y:s_y + win_radius_y, s_x - win_radius_x:s_x + win_radius_x])

                shift, _, _ = phase_cross_correlation(mas_subset, sl_subset, upsample_factor=100)
                c_shift_x = s_x - m_x
                c_shift_y = s_y - m_y
                result.loc[len(result)] = [m_x, m_y, c_shift_x-shift[1], c_shift_y-shift[0]]
                if output is not None:
                    output('Estimating Subpixel Shifts:', len(result), points, f' - {c_shift_x-shift[1]}, {c_shift_y-shift[0]}')
                if len(result) >= points:
                    break

    return result

def parameter_estimation(shifts, degree:int = 2):

    coords = shifts[['masterX', 'masterY', 'shiftX', 'shiftY']].to_numpy()

    print(f'Mean: {np.mean(coords[:,2])}, {np.mean(coords[:,3])}')

    ransac_pipeline_dx = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.001,  # Tune based on noise level
            max_trials=10000
        )
    )

    ransac_pipeline_dy = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.001,
            max_trials=10000
        )
    )

    # Fit models
    ransac_pipeline_dx.fit(coords[:,0:2], coords[:,2])
    ransac_pipeline_dy.fit(coords[:,0:2], coords[:,3])

    return ransac_pipeline_dx, ransac_pipeline_dy


