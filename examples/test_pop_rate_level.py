import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    stimulus,
    inner_hair_cell,
    map_to_synapse,
    synapse,
    SynapseMapping,
    Species,
    NoiseType,
    PowerLaw,
    FiberType,
    generate_an_population,
    Fiber,
)

def compute_population_rates():
    num_cfs = 1
    cfs = [8000.0]  # in Hz

    num_sponts = [20, 20, 60]  # LSR, MSR, HSR
    fibers_low, fibers_med, fibers_high = generate_an_population(num_cfs, *num_sponts)
    all_fibers = fibers_low + fibers_med + fibers_high

    stim_dbs = np.arange(-10, 105, 5)
    num_stims = len(stim_dbs)
    rates = np.zeros((num_cfs, len(all_fibers), num_stims))

    # Model parameters
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    implnt = PowerLaw.APPROXIMATED
    noise_type = NoiseType.RANDOM
    expliketype = SynapseMapping.SOFTPLUS
    fs = int(100e3)

    # Stimulus parameters
    T = 50e-3
    rt = 2.5e-3
    nrep = 100
    t = np.arange(0, T, 1/fs)
    irpts = int(rt * fs)

    for cf_idx, cf in enumerate(cfs):
        f0 = cf

        for fiber_idx, fiber in enumerate(all_fibers):
            spont = fiber.spont
            tabs = fiber.tabs
            trel = fiber.trel

            for stim_idx, stim_db in enumerate(stim_dbs):
                print(f"CF {cf_idx+1}/{num_cfs}, Fiber {fiber_idx+1}/{len(all_fibers)}, Stim {stim_idx+1}/{num_stims}")

                signal = np.sqrt(2) * 20e-6 * 10**(stim_db/20) * np.sin(2*np.pi*f0*t)
                signal[:irpts] *= np.linspace(0, 1, irpts)
                signal[-irpts:] *= np.linspace(1, 0, irpts)

                stim = stimulus.Stimulus(signal, fs, 2*T)

                vihc = inner_hair_cell(
                    stimulus=stim,
                    cf=cf,
                    n_rep=nrep,
                    cohc=cohc,
                    cihc=cihc,
                    species=species,
                )
                ihc_mapped = map_to_synapse(
                    ihc_output=vihc,
                    spontaneous_firing_rate=spont,
                    characteristic_frequency=cf,
                    time_resolution=stim.time_resolution,
                    mapping_function=expliketype,
                )

                syn = synapse(
                    amplitude_ihc=ihc_mapped,
                    cf=cf,
                    n_rep=nrep,
                    n_timesteps=stim.n_simulation_timesteps,
                    time_resolution=stim.time_resolution,
                    spontaneous_firing_rate=spont,
                    abs_refractory_period=tabs,
                    rel_refractory_period=trel,
                    noise=noise_type,
                    pla_impl=implnt,
                )

                psth = np.array(syn.psth)
                psth_bin_width = 0.5e-3
                bin_size = int(round(psth_bin_width * fs))
                num_bins = len(psth) // bin_size
                reshaped = np.reshape(psth[:num_bins * bin_size], (bin_size, num_bins))
                spike_rate = np.sum(reshaped, axis=0) / psth_bin_width / nrep

                ronset = int(round(1.5e-3 / psth_bin_width)) + 1
                roffset = int(round(T / psth_bin_width))

                rates[cf_idx, fiber_idx, stim_idx] = np.mean(spike_rate[ronset:ronset+roffset])

    # Plot
    plt.figure()
    for idx, fiber in enumerate(all_fibers):
        color = {
            FiberType.LOW: 'r',
            FiberType.MEDIUM: 'b',
            FiberType.HIGH: 'm'
        }.get(fiber.type, 'k')
        plt.plot(stim_dbs, rates[0, idx, :], color=color)

    plt.xlabel('Stimulus Level (dB SPL)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.title('Population Rate-Level Functions')
    plt.tight_layout()
    plt.show()

def main():
    compute_population_rates()

if __name__ == "__main__":
    main()
