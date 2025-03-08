
from pysar import tools
from pysar.insar import flat_interferogram, coherence, resampled_pair

if __name__ == "__main__":

    file_pair = ''
    output_path = ''

    pair = resampled_pair.ResampledPair(file_pair)
    phase_model, poly = flat_interferogram.get_flat_phase_model(pair)

    flat_phase = flat_interferogram.create_flattened_interferogram(pair, phase_model, poly)
    coh_data = coherence.compute_coherence(pair.master.slcdata.read(), pair.slave.slcdata.read(), flat_interfero=flat_phase, output=tools.output_console())

    coh = coherence.createCoherence(pair.master.metadata, pair.slave.metadata, coh_data)
    coh.save(directory=output_path)


