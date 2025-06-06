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


def compute_siicc(isi: np.ndarray) -> tuple[float, float]:
    """
    Computes the Short-term Inter-Spike Interval Correlation Coefficient (SIICC).
    
    Parameters:
        isi (np.ndarray): Array of inter-spike intervals in seconds.
        
    Returns:
        mean_isi (float): Mean inter-spike interval.
        siicc (float): Normalized lag-1 autocorrelation of ISIs.
    """
    if len(isi) < 2:
        return np.nan, np.nan

    isi_mean = np.mean(isi)
    isi_diff = isi - isi_mean
    numerator = np.sum(isi_diff[:-1] * isi_diff[1:])
    denominator = np.sum(isi_diff[:-1] ** 2)

    siicc = numerator / (denominator + np.finfo(float).eps)
    return isi_mean, siicc


def run_siicc_analysis():
    # === Model fiber parameters ===
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    mapping = SynapseMapping.SOFTPLUS

    # === Simulation parameters ===
    fs = int(100e3)
    stim_duration = 20  # seconds
    n_trials = 50
    psth_bin_width = 0.5e-3
    nrep = 1

    # === CFs and AN population ===
    num_cfs = 30
    cfs = np.logspace(np.log10(300), np.log10(20e3), num_cfs)
    n_fibers = [0, 10, 30]

    print("Generating AN fiber population...")
    ls, ms, hs = generate_an_population(num_cfs, *n_fibers)
    all_fibers = ls + ms + hs

    # === Stimulus ===
    stimulus = np.zeros(int(stim_duration * fs))

    m_vals = []
    p_vals = []
    trial = 0
    rng = np.random.default_rng()

    while trial < n_trials:
        print(f"Trial {trial + 1}/{n_trials}", end="")

        # Select random fiber
        cf_idx = rng.integers(0, num_cfs)
        fiber_idx = rng.integers(0, len(all_fibers) // num_cfs)
        fiber = all_fibers[cf_idx * (len(all_fibers) // num_cfs) + fiber_idx]
        cf = cfs[cf_idx]

        ihc_mapped = map_to_synapse(
            ihc_output=stimulus,
            spontaneous_firing_rate=fiber.spont,
            characteristic_frequency=cf,
            time_resolution=1 / fs,
            mapping_function=mapping,
        )

        syn = synapse(
            amplitude_ihc=ihc_mapped,
            cf=cf,
            n_rep=nrep,
            n_timesteps=len(stimulus),
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
            print(f": only {len(spike_times)} spikes - retrying")
            continue

        isi = np.diff(spike_times)

        if len(isi) > 500:
            mean_isi, siicc_val = compute_siicc(isi)
            m_vals.append(mean_isi)
            p_vals.append(siicc_val)
            print(f": {len(spike_times)} spikes")
            trial += 1
        else:
            print(f": only {len(spike_times)} spikes - retrying")

    # === Plot ===
    plt.figure()
    plt.plot(np.array(m_vals) * 1e3, p_vals, ".")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlim([0, 50])
    plt.ylim([-0.2, 0.2])
    plt.xlabel("Mean ISI (ms)")
    plt.ylabel("SIICC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_siicc_analysis()
