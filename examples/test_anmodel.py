"""
Replicates testANmodel_BEZ2018a.m using the brucezilany package.
"""

import numpy as np
import matplotlib.pyplot as plt

from brucezilany import (
    stimulus as stimgen,
    inner_hair_cell,
    synapse,
    map_to_synapse,
    SynapseMapping,
    NoiseType,
    PowerLaw,
    Species,
)


def run_anmodel():
    # --- Model Parameters ---
    cf = 1000  # Characteristic frequency (Hz)
    spont = 100  # Spontaneous firing rate (spikes/s)
    tabs = 0.6e-3  # Absolute refractory period (s)
    trel = 0.6e-3  # Mean relative refractory period (s)
    cohc = 1.0  # Normal OHC function
    cihc = 1.0  # Normal IHC function
    species = Species.CAT
    noise_type = NoiseType.RANDOM
    powerlaw_impl = PowerLaw.APPROXIMATED
    mapping_function = SynapseMapping.SOFTPLUS

    # --- Stimulus Parameters ---
    stim_db = 60
    fs = int(100e3)
    duration = 50e-3
    rt = 2.5e-3
    delay = 10e-3
    sim_dur = duration + delay + 2e-3  # margin for processing

    stim = stimgen.ramped_sine_wave(
        duration=duration,
        simulation_duration=4 * sim_dur,
        sampling_rate=fs,
        rt=rt,
        delay=delay,
        f0=cf,
        db=stim_db,
    )

    # --- Model Processing ---
    ihc_out = inner_hair_cell(
        stimulus=stim,
        cf=cf,
        n_rep=100,
        cohc=cohc,
        cihc=cihc,
        species=species,
    )

    mapped_ihc = map_to_synapse(
        ihc_output=ihc_out,
        spontaneous_firing_rate=spont,
        characteristic_frequency=cf,
        time_resolution=stim.time_resolution,
        mapping_function=mapping_function,
    )

    syn_out = synapse(
        amplitude_ihc=mapped_ihc,
        cf=cf,
        n_rep=100,
        n_timesteps=stim.n_simulation_timesteps,
        time_resolution=stim.time_resolution,
        spontaneous_firing_rate=spont,
        abs_refractory_period=tabs,
        rel_refractory_period=trel,
        noise=noise_type,
        pla_impl=powerlaw_impl,
    )

    return stim, ihc_out, syn_out


def plot_all(stim, ihc_out, syn_out):
    fs = stim.sampling_rate
    t_signal = np.arange(len(stim.data)) / fs
    t_model = np.arange(syn_out.n_timesteps) * stim.time_resolution

    nrep = syn_out.n_rep
    bin_width = 1e-4
    psthbins = int(bin_width * fs)
    Psth = np.sum(np.reshape(syn_out.psth, (-1, psthbins)), axis=1) / (nrep * bin_width)
    tvect = np.arange(len(Psth)) * bin_width

    Sout = np.mean(np.reshape(syn_out.synaptic_output, (nrep, -1)), axis=0)
    T_rd = np.mean(np.reshape(syn_out.redocking_time, (nrep, -1)), axis=0)
    T_rel = np.mean(np.reshape(syn_out.mean_relative_refractory_period, (nrep, -1)), axis=0)
    meanrate = np.array(syn_out.mean_firing_rate)
    varrate = np.array(syn_out.variance_firing_rate)

    meanrate = np.mean(np.reshape(meanrate, (nrep, -1)), axis=0)
    varrate = np.mean(np.reshape(varrate, (nrep, -1)), axis=0)

    # --- Plots ---
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_signal * 1e3, stim.data)
    plt.title("Acoustic Stimulus")
    plt.ylabel("Pressure (Pa)")
    plt.xlabel("Time (ms)")

    plt.subplot(3, 1, 2)
    plt.plot(t_signal * 1e3, np.array(ihc_out[:len(t_signal)]) * 1e3)
    plt.title("IHC Relative Membrane Potential")
    plt.ylabel("V_ihc (mV)")
    plt.xlabel("Time (ms)")

    plt.subplot(3, 1, 3)
    plt.bar(tvect * 1e3, Psth, width=bin_width * 1e3)
    plt.title("PSTH")
    plt.ylabel("Firing Rate (/s)")
    plt.xlabel("Time (ms)")
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_model * 1e3, Sout)
    plt.title("Mean Synaptic Release Rate")
    plt.ylabel("S_out (/s)")
    plt.xlabel("Time (ms)")

    plt.subplot(3, 1, 2)
    plt.plot(t_model * 1e3, meanrate)
    plt.title("Mean of Spike Rate")
    plt.ylabel("Mean Rate (/s)")
    plt.xlabel("Time (ms)")

    plt.subplot(3, 1, 3)
    plt.plot(t_model * 1e3, varrate)
    plt.title("Variance in Spike Rate")
    plt.ylabel("Var Rate (/s)")
    plt.xlabel("Time (ms)")
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_model * 1e3, Sout)
    plt.title("Synapse Output Rate")
    plt.ylabel("S_out (/s)")
    plt.xlabel("Time (ms)")

    plt.subplot(3, 1, 2)
    plt.plot(t_model * 1e3, T_rd * 1e3)
    plt.title("Mean Redocking Time")
    plt.ylabel("τ_rd (ms)")
    plt.xlabel("Time (ms)")

    plt.subplot(3, 1, 3)
    plt.plot(t_model * 1e3, T_rel * 1e3)
    plt.title("Mean Relative Refractory Period")
    plt.ylabel("t_rel (ms)")
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()


def main():
    stim, ihc_out, syn_out = run_anmodel()
    plot_all(stim, ihc_out, syn_out)


if __name__ == "__main__":
    main()
