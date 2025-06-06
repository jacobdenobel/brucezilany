import numpy as np
import matplotlib.pyplot as plt

from brucezilany import (
    generate_an_population,
    inner_hair_cell,
    map_to_synapse,
    synapse,
    stimulus,
    SynapseMapping,
    Species,
    NoiseType,
    PowerLaw
)


def find_cf_threshold_bez2018a(cf, fs, cohc, cihc, species,
                                noise_type, implnt, spont, tabs, trel,
                                expliketype=SynapseMapping.SOFTPLUS):
    """
    Estimate auditory nerve fiber threshold by increasing tone level
    until average firing rate exceeds spont + 10 spikes/s.
    """
    stimdb = -10
    f0 = cf
    nrep = 5
    T = 0.050  # 50 ms
    rt = 2.5e-3
    dt = 1 / fs

    t = np.arange(0, T, dt)
    mxpts = len(t)
    irpts = int(round(rt * fs))
    psth_bin_width = 0.5e-3
    psth_bins = int(round(psth_bin_width * fs))

    # Initialization
    firing_rate_increased_to = spont

    while firing_rate_increased_to < spont + 10 and stimdb < 50:
        stimdb += 1

        amp = np.sqrt(2) * 20e-6 * 10**(stimdb / 20)
        signal = amp * np.sin(2 * np.pi * f0 * t)
        # Ramp on
        signal[:irpts] *= np.linspace(0, 1, irpts)
        # Ramp off
        signal[-irpts:] *= np.linspace(1, 0, irpts)

        # Wrap into stimulus structure
        stim = stimulus.Stimulus(signal, fs, 2 * T)

        # IHC stage
        vihc = inner_hair_cell(
            stimulus=stim,
            cf=cf,
            n_rep=nrep,
            cohc=cohc,
            cihc=cihc,
            species=species
        )

        ihc_mapped = map_to_synapse(
            ihc_output=vihc,
            spontaneous_firing_rate=spont,
            characteristic_frequency=cf,
            time_resolution=stim.time_resolution,
            mapping_function=expliketype,
        )

        # Synapse stage
        syn_out = synapse(
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

        psth = np.array(syn_out.psth)
        n_bins = len(psth) // psth_bins
        binned = np.reshape(psth[:n_bins * psth_bins], (psth_bins, n_bins))
        spike_counts = np.sum(binned, axis=0) / nrep
        rates = spike_counts / psth_bin_width

        ronset = int(round(1.5e-3 / psth_bin_width)) + 1
        roffset = n_bins

        spontaneous_rate = np.mean(rates[roffset:])
        firing_rate_increased_to = np.mean(rates[ronset:ronset + roffset])

    return stimdb



def run_threshold_vs_cf():
    num_cfs = 20
    n_low, n_med, n_high = 0, 0, 15
    fs = int(100e3)
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    expliketype = SynapseMapping.SOFTPLUS

    cfs = np.logspace(np.log10(125), np.log10(15e3), num_cfs)
    thresholds = np.zeros((num_cfs, n_high))

    # Generate fibers
    low, med, high = generate_an_population(num_cfs, n_low, n_med, n_high)

    for cf_idx, cf in enumerate(cfs):
        print(f"CF {cf_idx + 1}/{num_cfs} - {cf:.1f} Hz")
        for fiber_idx in range(n_high):
            fiber = high[cf_idx * fiber_idx]
            print(f"  Fiber {fiber_idx + 1}/{n_high} (spont: {fiber.spont:.1f})")

            stimdb = find_cf_threshold_bez2018a(
                cf=cf,
                fs=fs,
                cohc=cohc,
                cihc=cihc,
                species=species,
                noise_type=noise_type,
                implnt=implnt,
                spont=fiber.spont,
                tabs=fiber.tabs,
                trel=fiber.trel,
                expliketype=expliketype
            )

            thresholds[cf_idx, fiber_idx] = stimdb

    # === Summary stats ===
    threshold_mean = np.mean(thresholds, axis=1)
    threshold_std = np.std(thresholds, axis=1)

    # === Plot ===
    plt.figure()
    for i in range(n_high):
        plt.semilogx(cfs / 1e3, thresholds[:, i], 'kx')
    plt.errorbar(cfs / 1e3, threshold_mean, yerr=threshold_std, fmt='r.-', label='Mean ± STD')
    plt.xlabel("CF (kHz)")
    plt.ylabel("Threshold (dB SPL)")
    plt.title("Auditory Nerve Threshold vs CF (HS fibers)")
    plt.xlim([0.1, 20])
    plt.xticks([0.1, 1, 10], labels=["0.1", "1", "10"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_threshold_vs_cf()
