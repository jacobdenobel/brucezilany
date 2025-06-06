import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    synapse, 
    Species, 
    NoiseType, 
    PowerLaw, 
    SynapseMapping, 
    map_to_synapse, 
    generate_an_population,
    set_seed
)


def run_spontaneous_rate_analysis():
    # === Model parameters ===
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    mapping = SynapseMapping.SOFTPLUS

    fs = int(100e3)
    duration = 30.0  # seconds
    nrep = 1
    trials = 738

    # === AN population ===
    numcfs = 30
    cfs = np.logspace(np.log10(300), np.log10(20e3), numcfs)
    n_low, n_med, n_high = 10, 10, 30
    low, med, high = generate_an_population(numcfs, n_low, n_med, n_high)
    fibers = low + med + high

    # === Time vector and empty stimulus ===
    t = np.arange(0, duration, 1 / fs)
    silence = np.zeros_like(t)

    spont_rates = np.zeros(trials)
    mean_isis = np.zeros(trials)

    trial = 0
    while trial < trials:
        print(f"Trial {trial + 1}/{trials}")

        cf_idx = np.random.randint(0, numcfs)
        fiber_idx = np.random.randint(0, len(fibers) // numcfs)
        fiber = fibers[cf_idx * (n_low + n_med + n_high) + fiber_idx]
        cf = cfs[cf_idx]

        ihc_mapped = map_to_synapse(
            ihc_output=silence,
            spontaneous_firing_rate=fiber.spont,
            characteristic_frequency=cf,
            time_resolution=1 / fs,
            mapping_function=mapping,
        )
        set_seed(trial * 7)
        syn = synapse(
            amplitude_ihc=ihc_mapped,
            cf=cf,
            n_rep=nrep,
            n_timesteps=len(silence),
            time_resolution=1 / fs,
            spontaneous_firing_rate=fiber.spont,
            abs_refractory_period=fiber.tabs,
            rel_refractory_period=fiber.trel,
            noise=noise_type,
            pla_impl=implnt,
        )

        psth = np.array(syn.psth)
        spike_times = np.where(psth == 1)[0] / fs
        if len(spike_times) < 2:
            print("Not enough spikes, retrying...")
            continue

        isi = np.diff(spike_times)
        mean_isis[trial] = np.mean(isi)
        spont_rates[trial] = fiber.spont
        trial += 1

    # === Histogram plotting ===
    edges = np.arange(0, 121, 1)
    hist = np.histogram(1 / mean_isis, bins=edges)[0]

    plt.figure()
    plt.bar(edges[:-1], hist, width=1)
    plt.xlabel("Spontaneous Rate (/s)")
    plt.ylabel("Number of Units")
    plt.xlim([-0.5, 120])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_spontaneous_rate_analysis()
