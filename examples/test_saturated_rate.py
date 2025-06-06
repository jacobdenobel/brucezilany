import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    inner_hair_cell,
    synapse,
    stimulus as stimgen,
    map_to_synapse,
    Species,
    NoiseType,
    PowerLaw,
    SynapseMapping,
    generate_an_population,
    find_cf_threshold
)



def compute_saturation_rates():
    # === Parameters ===
    num_cfs = 5
    cfs = np.logspace(np.log10(300), np.log10(20e3), num_cfs)
    fs = int(100e3)
    stim_duration = 50e-3
    rise_fall_time = 2.5e-3
    psth_bin_width = 0.5e-3
    nrep = 10

    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    mapping = SynapseMapping.SOFTPLUS

    num_sponts = 4
    num_stims = 55

    fibers = generate_an_population(
        n_cf=num_cfs,
        n_low=0,
        n_med=0,
        n_high=num_sponts,
    )[2]  # High spontaneous fibers only

    rates = np.zeros((num_cfs, num_sponts, num_stims))
    t = np.arange(0, stim_duration, 1 / fs)
    irpts = int(rise_fall_time * fs)

    for cf_idx, cf in enumerate(cfs):
        f0 = cf

        for fiber_idx in range(num_sponts):
            fiber = fibers[cf_idx * num_sponts + fiber_idx]

            spont = fiber.spont
            tabs = fiber.tabs
            trel = fiber.trel

            threshold = find_cf_threshold(
                cf=cf,
                fs=fs,
                cohc=cohc,
                cihc=cihc,
                species=species,
                noise=noise,
                implnt=implnt,
                spont=spont,
                tabs=tabs,
                trel=trel,
                mapping=mapping,
            )

            stim_dbs = np.arange(threshold, threshold + num_stims)

            for stim_idx, stim_db in enumerate(stim_dbs):
                print(f"CF {cf_idx + 1}/{num_cfs}, Fiber {fiber_idx + 1}/{num_sponts}, Stim {stim_idx + 1}/{num_stims}")

                signal = np.sqrt(2) * 20e-6 * 10 ** (stim_db / 20) * np.sin(2 * np.pi * f0 * t)
                signal[:irpts] *= np.linspace(0, 1, irpts)
                signal[-irpts:] *= np.linspace(1, 0, irpts)

                stim = stimgen.Stimulus(signal, fs, 2 * stim_duration)

                vihc = inner_hair_cell(stim, cf, nrep, cohc, cihc, species)
                ihc_mapped = map_to_synapse(
                    ihc_output=vihc,
                    spontaneous_firing_rate=spont,
                    characteristic_frequency=cf,
                    time_resolution=stim.time_resolution,
                    mapping_function=mapping,
                )
                syn = synapse(
                    amplitude_ihc=ihc_mapped,
                    cf=cf,
                    n_rep=nrep,
                    n_timesteps=len(signal),
                    time_resolution=1 / fs,
                    spontaneous_firing_rate=spont,
                    abs_refractory_period=tabs,
                    rel_refractory_period=trel,
                    noise=noise,
                    pla_impl=implnt,
                )

                psth = syn.psth
                binsize = int(round(psth_bin_width * fs))
                pr = np.sum(np.reshape(psth[:len(psth) // binsize * binsize], (binsize, -1)), axis=0) / nrep
                rate = pr / psth_bin_width

                onset_idx = int(round(1.5e-3 / psth_bin_width))
                offset_idx = int(round(stim_duration / psth_bin_width))
                rates[cf_idx, fiber_idx, stim_idx] = np.mean(rate[onset_idx:offset_idx])


    # === Plotting ===
    plt.figure()
    plt.semilogx(cfs / 1e3, np.max(rates, axis=2), 'ko')
    plt.xlim([0.1, 40])
    plt.ylim([100, 350])
    plt.xlabel("Characteristic Frequency (kHz)")
    plt.ylabel("Discharge Rate at Saturation (/s)")
    plt.title("Saturation Rates Across CFs")
    plt.tight_layout()
    plt.show()


def main():
    compute_saturation_rates()


if __name__ == "__main__":
    main()
