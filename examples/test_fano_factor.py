import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    stimulus as stimgen,
    inner_hair_cell,
    synapse,
    set_seed,
    Species,
    NoiseType,
    PowerLaw,
    SynapseMapping,
    map_to_synapse,
)


def compute_fano_factors():
    # === Model fiber parameters ===
    cf = 1500  # Characteristic frequency in Hz
    spont = 50
    tabs = 0.6e-3
    trel = 0.6e-3
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    expliketype = SynapseMapping.SOFTPLUS

    # === Stimulus parameters ===
    fs = int(100e3)
    T = 25  # Duration in seconds
    rt = 2.5e-3
    stim_db = -np.inf  # Silence for spontaneous activity
    f0 = cf

    dt = 1 / fs
    num_ts = 14
    trials = 10
    Ts = np.logspace(np.log10(1e-3), np.log10(10), num=num_ts)
    Ts = np.round(Ts / dt) * dt

    # === Stimulus generation ===
    stim = stimgen.ramped_sine_wave(
        duration=T,
        simulation_duration=2 * T,
        sampling_rate=fs,
        rt=rt,
        delay=0,
        f0=f0,
        db=stim_db,
    )

    vihc = inner_hair_cell(
        stimulus=stim,
        cf=cf,
        n_rep=1,
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

    ft = np.zeros((trials, num_ts))
    ft_shuf = np.zeros((trials, num_ts))
    mean_rate = np.zeros((trials, num_ts))

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}")
        set_seed(trial * 11)

        syn_out = synapse(
            amplitude_ihc=ihc_mapped,
            cf=cf,
            n_rep=1,
            n_timesteps=stim.n_simulation_timesteps,
            time_resolution=stim.time_resolution,
            spontaneous_firing_rate=spont,
            abs_refractory_period=tabs,
            rel_refractory_period=trel,
            noise=noise_type,
            pla_impl=implnt,
        )

        psth = np.array(syn_out.psth)
        sim_time = len(psth) / fs
        tvect = np.arange(len(psth)) / fs

        spike_times = tvect[psth.astype(bool)]
        isis = np.diff(spike_times)

        if len(isis) == 0:
            continue

        shuffled_isis = np.random.permutation(isis)
        shuffled_spike_times = np.cumsum(shuffled_isis)
        shuffled_spike_times = shuffled_spike_times[shuffled_spike_times < sim_time]

        shuffled_psth, _ = np.histogram(shuffled_spike_times, bins=len(psth), range=(0, sim_time))

        for j, T_val in enumerate(Ts):
            binsize = int(round(T_val * fs))
            n_bins = len(psth) // binsize

            if n_bins == 0:
                continue

            reshaped_psth = np.reshape(psth[:n_bins * binsize], (binsize, n_bins))
            reshaped_shuf = np.reshape(shuffled_psth[:n_bins * binsize], (binsize, n_bins))

            counts = np.sum(reshaped_psth, axis=0)
            counts_shuf = np.sum(reshaped_shuf, axis=0)

            ft[trial, j] = np.var(counts) / (np.mean(counts) + np.finfo(float).eps)
            ft_shuf[trial, j] = np.var(counts_shuf) / (np.mean(counts_shuf) + np.finfo(float).eps)
            mean_rate[trial, j] = np.mean(counts) / T_val

    # === Plotting ===
    plt.figure()
    plt.loglog(Ts * 1e3, ft.T, alpha=0.5)
    plt.loglog(Ts * 1e3, np.mean(ft, axis=0), 'k-', linewidth=2, label='Mean F(T)')
    plt.loglog(Ts * 1e3, ft_shuf.T, '--', alpha=0.4)
    plt.loglog(Ts * 1e3, np.mean(ft_shuf, axis=0), 'k--', linewidth=2, label='Mean F(T) Shuffled')
    plt.xlabel("T (ms)")
    plt.ylabel("F(T)")
    plt.xlim([1e0, 1e4])
    plt.ylim([0.2, 10])
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    compute_fano_factors()


if __name__ == "__main__":
    main()