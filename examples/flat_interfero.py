from pysar import tools
from pysar.insar import flat_interferogram, resampled_pair
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file_pair = '/home/timo/Data/pysar_test/TSX-1_0_2010-02-22__TSX-1_2010-03-05.pysar.resampled.xml'
    output_path = '/home/timo/Data/pysar_test/'

    pair = resampled_pair.ResampledPair(file_pair)
    phase_model, poly = flat_interferogram.get_flat_phase_model(pair)

    master_shape = (pair.master.metadata.number_rows, pair.master.metadata.number_columns)  # (height, width)

    # Generate sample data for x and y coordinates
    x = np.linspace(0, master_shape[1] - 1, int(master_shape[1] / 100))
    y = np.linspace(0, master_shape[0] - 1, int(master_shape[0] / 100))
    xx, yy = np.meshgrid(x, y)
    X = np.column_stack((xx.ravel(), yy.ravel()))
    point_poly = poly.transform(X)

    pred_phase = phase_model.predict(point_poly).reshape(xx.shape)
    I = np.exp(1j * pred_phase)  # Example complex interferogram
    pred_wrapped_phase = np.angle(I)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # Plot the estimated dx shifts
    im1 = ax1.imshow(pred_wrapped_phase, extent=[0, pair.master.metadata.number_columns, 0, pair.master.metadata.number_rows], origin='lower', cmap='hsv')
    ax1.set_title('Estimated Topographic Phase')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, label='phase')

    plt.tight_layout()
    plt.show()

    flat_phase = flat_interferogram.create_flattened_interferogram(pair, phase_model, poly, tools.output_console)

    flatty = flat_interferogram.createFlatInterferogram(pair.master.metadata, pair.slave.metadata, flat_phase)
    flatty.save(directory=output_path)