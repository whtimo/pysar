from pysar import slc, metadata, baseline, insar_pair, coregistration
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    file_pair = '/Users/timo/Documents/WuhanEast/bzar/pair/TDX-1_0_2018-07-24__TDX-1_0_2018-08-26.pair.xml'
    shifts_file = '/Users/timo/Documents/WuhanEast/pysar/shifts_bzar.csv'

    pair = insar_pair.fromBzarXml(file_pair)

    print(f'Baseline calculation: {pair.perpendicular_baseline} m Perpendicular Baseline and a Temporal Baseline of {pair.temporal_baseline} days')

    shifts = coregistration.subpixel_shifts(pair.master.metadata, pair.slave.metadata, int(pair.shift_x), int(pair.shift_y), pair.master.slcdata.read(), pair.slave.slcdata.read())
    shifts.to_csv(shifts_file, index=True)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot for shiftX
    sc1 = ax1.scatter(shifts['masterX'], shifts['masterY'], c=shifts['shiftX'], cmap='viridis')
    ax1.set_title('Shift in X')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(sc1, ax=ax1, label='shiftX')

    # Scatter plot for shiftY
    sc2 = ax2.scatter(shifts['masterX'], shifts['masterY'], c=shifts['shiftY'], cmap='viridis')
    ax2.set_title('ShiftY in Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(sc2, ax=ax2, label='shiftY')

    # Show the plot
    plt.tight_layout()
    plt.show()

    master_shape = (pair.master.metadata.number_rows, pair.master.metadata.number_columns)  # (height, width)

    # Generate sample data for x and y coordinates
    x = np.linspace(0, master_shape[1] - 1, int(master_shape[1] / 100))
    y = np.linspace(0, master_shape[0] - 1, int(master_shape[0] / 100))
    xx, yy = np.meshgrid(x, y)
    X = np.column_stack((xx.ravel(), yy.ravel()))

    est_dx_1, est_dy_1 = coregistration.parameter_estimation(shifts, 1)
    pred_dx1 = est_dx_1.predict(X).reshape(xx.shape)
    pred_dy1 = est_dy_1.predict(X).reshape(yy.shape)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the estimated dx shifts
    im1 = ax1.imshow(pred_dx1, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='viridis', vmin=np.min(pred_dx1) - 0.01, vmax=np.max(pred_dx1) + 0.01)
    ax1.set_title('Estimated Shifts (dx) with degree = 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, label='dx')

    # Plot the estimated dy shifts
    im2 = ax2.imshow(pred_dy1, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='viridis', vmin=np.min(pred_dy1) - 0.01, vmax=np.max(pred_dy1) + 0.01)
    ax2.set_title('Estimated Shifts (dy) with degree = 1')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(im2, ax=ax2, label='dy')

    # Show the plot
    plt.tight_layout()
    plt.show()

    est_dx_2, est_dy_2 = coregistration.parameter_estimation(shifts, 2)
    pred_dx2 = est_dx_2.predict(X).reshape(xx.shape)
    pred_dy2 = est_dy_2.predict(X).reshape(yy.shape)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the estimated dx shifts
    im1 = ax1.imshow(pred_dx2, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='viridis', vmin=np.min(pred_dx2) - 0.01, vmax=np.max(pred_dx2) + 0.01)
    ax1.set_title('Estimated Shifts (dx) with degree = 2')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, label='dx')

    # Plot the estimated dy shifts
    im2 = ax2.imshow(pred_dy2, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='viridis', vmin=np.min(pred_dy2) - 0.01, vmax=np.max(pred_dy2) + 0.01)
    ax2.set_title('Estimated Shifts (dy) with degree = 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(im2, ax=ax2, label='dy')

    # Show the plot
    plt.tight_layout()
    plt.show()




