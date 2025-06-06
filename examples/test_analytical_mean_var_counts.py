"""
Replicates testAnalyticalMeanVarCounts_BEZ2018a.m using the brucezilany package.
"""

import numpy as np
import matplotlib.pyplot as plt

from brucezilany import (
    stimulus as stimgen,
    inner_hair_cell,
    synapse,
    map_to_synapse,
    set_seed,
    Species,
    NoiseType,
    PowerLaw,
    SynapseMapping,
)

def run_variance_analysis():
    # Model parameters
    cf = 8000  # Characteristic frequency in Hz
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    spont = 100  # Spontaneous firing rate
    tabs = 0.6e-3
    trel = 0.6e-3
    implnt = PowerLaw.APPROXIMATED
    expliketype = SynapseMapping.SOFTPLUS

    # Stimulus parameters
    f0 = cf
    fs = int(100e3)
    T = 0.25  # Duration in seconds
    rt = 2.5e-3
    stim_db = 20
    ondelay = 25e-3

    # PSTH parameters
    nrep = 1
    psth_bin_widths = [5e-4, 5e-3, 5e-2]
    trials = 1000

    stim = stimgen.ramped_sine_wave(
        duration=T,
        simulation_duration=2 * T,
        sampling_rate=fs,
        rt=rt,
        delay=ondelay,
        f0=f0,
        db=stim_db,
    )


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

    psth_data = [[], [], []]
    meanrate_data = [[], [], []]
    varrate_data = [[], [], []]
    psth_times = []

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}")
        set_seed(trial * 7)
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

        for i, bin_width in enumerate(psth_bin_widths):
            binsize = int(round(bin_width * fs))
            n_bins = len(syn_out.psth) // binsize

            # Truncate PSTH and related arrays
            n_used = n_bins * binsize
            psth_trim = syn_out.psth[:n_used]
            meanrate_trim = syn_out.mean_firing_rate[:n_used]
            varrate_trim = syn_out.variance_firing_rate[:n_used]

            psth_binned = np.sum(np.reshape(psth_trim, (-1, binsize)), axis=1)
            mean_binned = np.mean(np.reshape(meanrate_trim, (-1, binsize)), axis=1)
            var_binned = np.mean(np.reshape(varrate_trim, (-1, binsize)), axis=1)

            psth_data[i].append(psth_binned)
            meanrate_data[i].append(mean_binned)
            varrate_data[i].append(var_binned)

            if trial == 0:
                psth_times.append(np.arange(n_bins) * bin_width)

    for i, bin_width in enumerate(psth_bin_widths):
        psth_arr = np.array(psth_data[i])  # shape: (trials, bins)
        mean_arr = np.array(meanrate_data[i])
        var_arr = np.array(varrate_data[i])
        time = psth_times[i]

        mean_count = np.mean(psth_arr, axis=0)
        var_count = np.var(psth_arr, axis=0)
        mean_theoretical = np.mean(mean_arr, axis=0) * bin_width
        var_theoretical = np.mean(var_arr, axis=0) * bin_width

        marker_size = 6 if bin_width > 10e-3 else 2

        plt.figure(figsize=(10, 6))

        # Top plot: mean spike counts
        plt.subplot(2, 1, 1)
        plt.bar(time, mean_count, width=bin_width, edgecolor='k', color='0.8')
        plt.plot(time + bin_width / 2, mean_theoretical, 'ro',
                markerfacecolor='r', markersize=marker_size)
        plt.ylabel("E[count]")
        plt.xlabel("Time (s)")
        plt.title(f"PSTH bin width = {bin_width * 1e3:.2f} ms")
        plt.xlim(time[0], time[-1])

        # Bottom plot: var spike counts
        plt.subplot(2, 1, 2)
        plt.bar(time, var_count, width=bin_width, edgecolor='k', color='0.8')
        plt.plot(time + bin_width / 2, var_theoretical, 'go',
                markerfacecolor='g', markersize=marker_size)
        plt.plot(time + bin_width / 2, mean_theoretical, 'bo',
                markerfacecolor='b', markersize=marker_size)
        plt.ylabel("var[count]")
        plt.xlabel("Time (s)")
        plt.xlim(time[0], time[-1])

        plt.tight_layout()
    plt.show()


def main():
    run_variance_analysis()


if __name__ == "__main__":
    main()
