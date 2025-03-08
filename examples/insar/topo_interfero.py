from pysar.insar import flat_interferogram, topo_interferogram
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":

    file_inf = ''
    srtm_path = ''
    output_path = ''

    interfero = flat_interferogram.FlatInterferogram(file_inf)

    topo_phases = topo_interferogram.get_topographic_phases(interfero.master_metadata,
                                                     interfero.slave_metadata,
                                                     srtm_path)


    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    im1 = ax1.imshow(np.angle(topo_phases), cmap='hsv', vmin=-3.14, vmax=3.14)
    ax1.set_title('Estimated Topographic Phase')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, label='phase')

    plt.tight_layout()
    plt.show()

    flat_phases = interfero.read()
    rem_phases = topo_interferogram.remove_topographic_phases(flat_phases, topo_phases)

    toppy = topo_interferogram.createTopoInterferogram(interfero.master_metadata, interfero.slave_metadata, rem_phases)
    toppy.save(directory=output_path)
