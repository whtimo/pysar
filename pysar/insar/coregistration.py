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
                #mas_subset = master_data[m_y-win_radius_y -search_size_y:m_y+win_radius_y+search_size_y,m_x-win_radius_x-search_size_x:m_x+win_radius_x+search_size_x]
                #sl_subset = slave_data[s_y+coarse_shift_y -win_radius_y -search_size_y:s_y+coarse_shift_y+win_radius_y+search_size_y,s_x+coarse_shift_x-win_radius_x-search_size_x:s_x+coarse_shift_x+win_radius_x+search_size_x]
                #shift, max_corr = subpixel_shift(abs(mas_subset), abs(sl_subset), search_size_x, search_size_y, upsample_factor)
                #if max_corr > corr_threshold:
                mas_subset = np.abs(master_data[m_y - win_radius_y:m_y + win_radius_y, m_x - win_radius_x:m_x + win_radius_x])
                sl_subset = np.abs(slave_data[s_y - win_radius_y:s_y + win_radius_y, s_x - win_radius_x:s_x + win_radius_x])
                #
                #shift, error, diffphase = phase_cross_correlation(mas_subset, sl_subset)

                # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                #
                # # Plot the estimated dx shifts
                # im1 = ax1.imshow(mas_subset, cmap='gray', vmin=0, vmax=256)
                #
                # # Plot the estimated dy shifts
                # im2 = ax2.imshow(sl_subset,cmap='gray', vmin=0, vmax=256)
                #
                #
                # # Show the plot
                # plt.tight_layout()
                # plt.show()

                shift, error, diffphase = phase_cross_correlation(mas_subset, sl_subset, upsample_factor=100)
                #print(f'Shift: {shift} - Error: {error} - Diffphase: {diffphase}')
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

    # dx_mean = np.mean(coords[:,2])
    # dx_std = np.std(coords[:,2])
    # dy_mean = np.mean(coords[:,3])
    # dy_std = np.std(coords[:,3])
    #
    # # Define the range for dx and dy
    # dx_range = (dx_mean - dx_std, dx_mean + dx_std)
    # dy_range = (dy_mean - dy_std, dy_mean + dy_std)
    #
    # # Filter the rows where shiftX is within dx_range and shiftY is within dy_range
    # filtered_coords = coords[
    #     (coords[:, 2] >= dx_range[0]) & (coords[:, 2] <= dx_range[1]) &
    #     (coords[:, 3] >= dy_range[0]) & (coords[:, 3] <= dy_range[1])
    #     ]
    #
    # print(f'Filtered Mean: {np.mean(filtered_coords[:, 2])}, {np.mean(filtered_coords[:, 3])}')

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

    ransac_pipeline_dx = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.001,  # Tune based on noise level
            max_trials=10000
        )
    )


    # Fit models
    ransac_pipeline_dx.fit(coords[:,0:2], coords[:,2])
    ransac_pipeline_dy.fit(coords[:,0:2], coords[:,3])



    return ransac_pipeline_dx, ransac_pipeline_dy


def parameter_estimation_test(shifts, degree: int = 2):
    coords = shifts[['masterX', 'masterY', 'shiftX', 'shiftY']].to_numpy()

    cont = True

    while cont:
        print(f'Mean: {np.mean(coords[:, 2])}, {np.mean(coords[:, 3])}')

        # ransac_pipeline_dx = make_pipeline(
        #     PolynomialFeatures(degree),
        #     RANSACRegressor(
        #         estimator=LinearRegression(),
        #         residual_threshold=0.001,  # Tune based on noise level
        #         max_trials=10000
        #     )
        # )
        # ransac_pipeline_dy = make_pipeline(
        #     PolynomialFeatures(degree),
        #     RANSACRegressor(
        #         estimator=LinearRegression(),
        #         residual_threshold=0.001,
        #         max_trials=10000
        #     )
        # )

        ransac_pipeline_dx = make_pipeline(
            PolynomialFeatures(degree),
            TheilSenRegressor()
        )
        ransac_pipeline_dy = make_pipeline(
            PolynomialFeatures(degree),
            TheilSenRegressor()

        )

        # Fit models
        ransac_pipeline_dx.fit(coords[:, 0:2], coords[:, 2])
        ransac_pipeline_dy.fit(coords[:, 0:2], coords[:, 3])

        sh_x = ransac_pipeline_dx.predict(coords[:, 0:2])
        sh_y = ransac_pipeline_dy.predict(coords[:, 0:2])
        dsx = np.abs(sh_x - coords[:, 2])
        dsy = np.abs(sh_y - coords[:, 3])
        dist = np.sqrt(dsx**2 + dsy**2)
        ix = np.argmin(dist)
        val = dist[ix]
        if val < 0.01:
            cont = False
            print(f'Finished with max dist: {val}. Using {len(coords)} points.')
        else:
            coords = np.delete(coords, ix, axis=0)
            print(f'Max dist: {val}. Now using {len(coords)} points.')

    return ransac_pipeline_dx, ransac_pipeline_dy