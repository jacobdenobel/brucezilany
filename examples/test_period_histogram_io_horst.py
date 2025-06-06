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


def generate_stimulus(fs, duration, f0, ramp_time):
    t = np.arange(0, duration, 1 / fs)
    signal = np.sqrt(2) * 20e-6 * np.sin(2 * np.pi * f0 * t)
    ramp_samples = int(ramp_time * fs)
    signal[:ramp_samples] *= np.linspace(0, 1, ramp_samples)
    signal[-ramp_samples:] *= np.linspace(1, 0, ramp_samples)
    return signal


def period_histogram_analysis(spontstr="HSR"):
    stimdbs = np.arange(-20, 61, 20)
    numdbs = len(stimdbs)

    fiber_map = {
        "HSR": (600, 591, 37),
        "MSR": (700, 700, 12),
        "LSR": (550, 550, 0.1),
    }
    mapping_func = SynapseMapping.SOFTPLUS
    if spontstr not in fiber_map:
        raise ValueError("Unknown spont rate class")

    f0, cf, spont = fiber_map[spontstr]
    fs = int(100e3)
    stimdur = 1.0
    restdur = 0.5
    sim_dur = stimdur + restdur
    nrep_stim = 400
    rt = 5e-3

    t_st = 10e-3
    t_end = np.floor((stimdur - 20e-3) / (1 / f0)) * (1 / f0) + t_st
    ind_st = int(round(t_st * fs))
    ind_end = int(round(t_end * fs))

    pin = generate_stimulus(fs, stimdur, f0, rt)

    num_phbins = 32
    phbinwidth = 1 / f0 / num_phbins

    stim_ph = np.zeros((numdbs, num_phbins))
    periodhistograms = np.zeros((numdbs, num_phbins))
    periodhistograms_scaled = np.zeros((numdbs, num_phbins))
    meanrates = np.zeros(numdbs)
    SIs = np.zeros(numdbs)
    numspikes = np.zeros(numdbs)
    phasediffs = np.zeros(numdbs)
    shiftvals = np.zeros(numdbs, dtype=int)

    fig_io = plt.figure()

    for dblp, stimdb in enumerate(stimdbs):
        print(f"Run {dblp + 1}/{numdbs}")

        stim_amp = pin * 10**(stimdb / 20)
        stim = stimulus.Stimulus(stim_amp, fs, sim_dur)
        vihc = inner_hair_cell(stim, cf, nrep_stim, cohc=1.0, cihc=1.0, species=Species.CAT)
        ihc_mapped = map_to_synapse(
            ihc_output=vihc,
            spontaneous_firing_rate=spont,
            characteristic_frequency=cf,
            time_resolution=stim.time_resolution,
            mapping_function=mapping_func,
        )
        syn = synapse(
            amplitude_ihc=ihc_mapped,
            cf=cf,
            n_rep=nrep_stim,
            n_timesteps=len(stim_amp),
            time_resolution=1 / fs,
            spontaneous_firing_rate=spont,
            abs_refractory_period=0.7e-3,
            rel_refractory_period=0.7e-3,
            noise=NoiseType.RANDOM,
            pla_impl=PowerLaw.APPROXIMATED,
        )

        psth = syn.psth
        meanrates[dblp] = np.sum(psth[ind_st:ind_end]) / ((t_end - t_st) * nrep_stim)

        psth_section = psth[ind_st:ind_end]
        tpsth = np.arange(len(psth_section)) / fs

        for lp, val in enumerate(psth_section):
            phase = (2 * np.pi * f0 * tpsth[lp]) % (2 * np.pi)
            phbin = int(round(phase / (2 * np.pi * f0 * phbinwidth))) % num_phbins
            periodhistograms[dblp, phbin] += val

        total_spikes = np.sum(periodhistograms[dblp])
        SI_sin = np.dot(periodhistograms[dblp], np.sin(2 * np.pi * np.arange(1, num_phbins + 1) / num_phbins)) / total_spikes
        SI_cos = np.dot(periodhistograms[dblp], np.cos(2 * np.pi * np.arange(1, num_phbins + 1) / num_phbins)) / total_spikes
        SIs[dblp] = np.sqrt(SI_sin**2 + SI_cos**2)
        numspikes[dblp] = total_spikes

        ph_ph = np.unwrap(np.angle(np.fft.fft(periodhistograms[dblp])))
        phasevals = np.linspace(0, 2 * np.pi, num_phbins, endpoint=False)

        stim_ph[dblp] = np.sqrt(2) * 1e3 * 20e-6 * 10**(stimdb / 20) * np.sin(phasevals)
        ph_st = np.unwrap(np.angle(np.fft.fft(stim_ph[dblp])))

        phasediffs[dblp] = ph_ph[1] - ph_st[1]
        shiftvals[dblp] = int(round(phasediffs[dblp] / (2 * np.pi / num_phbins)))

        periodhistograms_scaled[dblp] = periodhistograms[dblp] / nrep_stim / ((t_end - t_st) / num_phbins)

        fig = plt.figure()
        shifted = np.roll(periodhistograms_scaled[dblp], shiftvals[dblp])
        plt.bar(phasevals, shifted)
        plt.plot(phasevals, ((np.max(shifted) - np.min(shifted)) / 2) * (1 + np.sin(phasevals)) + np.min(shifted))
        plt.xlabel("Phase (rad)")
        plt.ylabel("Instantaneous Rate (spikes/s)")
        plt.title(f"{mapping_func}; {spontstr}; {stimdb} dB SPL; {int(numspikes[dblp])} spikes")

        plt.figure(fig_io.number)
        plt.scatter(stim_ph[dblp], shifted)
        plt.xlabel("Instantaneous Pressure (mPa)")
        plt.ylabel("Instantaneous Rate (spikes/s)")
        plt.title(f"{mapping_func}; {spontstr}")

    plt.figure(fig_io.number)
    plt.legend([f"{db} dB SPL" for db in stimdbs], loc='upper left')

    plt.figure()
    plt.plot(stimdbs, SIs, label="SI")
    plt.ylabel("SI")
    plt.xlabel("Stimulus Level (dB SPL)")
    plt.ylim(0, 1)

    plt.twinx()
    plt.plot(stimdbs, meanrates, 'r', label="Mean Rate")
    plt.ylabel("Mean Rate (spikes/s)")
    plt.title(f"{mapping_func}; {spontstr}")

    plt.show()


if __name__ == "__main__":
    period_histogram_analysis()
