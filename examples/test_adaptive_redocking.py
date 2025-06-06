"""
Replicates testAdaptiveRedocking_BEZ2018a.m using the brucezilany package.
"""

import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    stimulus,
    inner_hair_cell,
    map_to_synapse,
    synapse,
    SynapseMapping,
    NoiseType,
    PowerLaw,
    Species,
    set_seed
)


def run_trial(pla_output, cf, nrep, dt, n_timesteps, noise, impl, spont, tabs, trel):
    return synapse(
        amplitude_ihc=pla_output,
        cf=cf,
        n_rep=nrep,
        n_timesteps=n_timesteps,
        time_resolution=dt,
        spontaneous_firing_rate=spont,
        abs_refractory_period=tabs,
        rel_refractory_period=trel,
        noise=noise,
        pla_impl=impl,
    )


def main():
    # === Model parameters ===
    cf = 5_000
    spont = 100
    tabs = 0.6e-3
    trel = 0.6e-3
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    mapping_fn = SynapseMapping.SOFTPLUS

    # === Stimulus parameters ===
    stim_db = 60
    fs = int(100e3)
    duration = 0.25
    delay = 25e-3
    rt = 2.5e-3
    trials = 1000
    nrep = 1

    # === PSTH binning ===
    psth_bin_width = 5e-4
    psth_bins = int(psth_bin_width * fs)
    
    # === Stimulus generation ===
    stim = stimulus.ramped_sine_wave(
        duration=duration,
        simulation_duration=2.0 * duration,
        sampling_rate=fs,
        rt=rt,
        delay=delay,
        f0=cf,
        db=stim_db,
    )

    dt = stim.time_resolution
    n_timesteps = stim.n_simulation_timesteps

    # === Inner Hair Cell processing ===
    ihc = inner_hair_cell(
        stimulus=stim,
        cf=cf,
        n_rep=nrep,
        cohc=cohc,
        cihc=cihc,
        species=species,
    )

    # === Synapse Mapping ===

    # === Allocate output buffers ===
    num_bins = n_timesteps // psth_bins
    eb_stride = 500
    eb_bins = n_timesteps // eb_stride
    nmax = 50

    spike_counts = np.zeros((trials, num_bins))
    cum_counts = np.zeros_like(spike_counts)
    synout_vecs = np.zeros((nmax, n_timesteps))
    trd_vecs = np.zeros((nmax, eb_bins))
    trel_vecs = np.zeros((nmax, n_timesteps))
    trd_all = np.zeros((trials, eb_bins))
    spike_totals = np.zeros(trials)

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}")
        set_seed(trial * 7)
        pla = map_to_synapse(
            ihc_output=ihc,
            spontaneous_firing_rate=spont,
            characteristic_frequency=cf,
            time_resolution=dt,
            mapping_function=mapping_fn,
        )
        out = run_trial(pla, cf, nrep, dt, n_timesteps, noise_type, implnt, spont, tabs, trel)

        # PSTH binning
        psth = np.sum(np.reshape(out.psth, (-1, psth_bins)), axis=1)
        spike_counts[trial, :] = psth
        cum_counts[trial, :] = np.cumsum(psth)
        spike_totals[trial] = np.sum(out.psth)

        # Redocking extraction for errorbar plots
        trd_all[trial, :] = out.redocking_time[::eb_stride] * 1e3

        # Store a subset of all traces
        if trial < nmax:
            synout_vecs[trial, :] = out.synaptic_output
            trd_vecs[trial, :] = out.redocking_time[::eb_stride] * 1e3
            trel_vecs[trial, :] = np.array(out.mean_relative_refractory_period) * 1e3

    # === Time vectors ===
    t_psth = np.arange(num_bins) * psth_bin_width
    t_full = np.arange(n_timesteps) / fs
    t_eb = np.arange(eb_bins) * eb_stride / fs

    # === Plot PSTH ===
    plt.figure()
    mean_rate = np.mean(spike_counts, axis=0) / psth_bin_width
    plt.bar(t_psth, mean_rate, width=psth_bin_width)
    plt.title("PSTH")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (/s)")
    plt.tight_layout()

    # === Error bar plot for τ_rd ===
    trd_mean = np.mean(trd_all, axis=0)
    trd_std = np.std(trd_all, axis=0)

    plt.figure()
    plt.errorbar(t_eb, trd_mean, trd_std)
    plt.title("Mean Redocking Time")
    plt.xlabel("Time (s)")
    plt.ylabel("τ_rd (ms)")
    plt.tight_layout()

    # === Synapse stats plots ===
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_full, synout_vecs.T[:, :nmax])
    plt.title("Synaptic Output (First 50 Trials)")
    plt.ylabel("S_out (/s)")

    plt.subplot(3, 1, 2)
    plt.plot(t_eb, trd_vecs.T[:, :nmax])
    plt.title("Redocking Time (First 50 Trials)")
    plt.ylabel("τ_rd (ms)")

    plt.subplot(3, 1, 3)
    plt.plot(t_full, trel_vecs.T[:, :nmax])
    plt.title("Relative Refractory Period (First 50 Trials)")
    plt.ylabel("t_rel (ms)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
