import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    stimulus,
    inner_hair_cell,
    synapse,
    Species,
    NoiseType,
    PowerLaw,
    SynapseMapping,
    set_seed,
    map_to_synapse,
    generate_an_population
)


def compute_noise_response_statistics():
    num_cfs = 10
    cfs = np.logspace(np.log10(300), np.log10(20e3), num=num_cfs)

    num_sponts = [2, 2, 6]
    fs = int(100e3)
    stim_duration = 2.0
    rt = 10e-3
    psth_bin_width = 1.25e-3
    psth_bins = int(round(psth_bin_width * fs))
    trials = 50
    stim_dbs = np.arange(0, 65, 5)
    num_stims = len(stim_dbs)

    cnt_noise = np.zeros((num_cfs, sum(num_sponts), num_stims, trials))

    # generate population
    fibers_LS, fibers_MS, fibers_HS = generate_an_population(num_cfs, *num_sponts)
    fibers_all = fibers_LS + fibers_MS + fibers_HS

    for cf_idx, cf in enumerate(cfs):
        for fiber_idx, fiber in enumerate(fibers_all[cf_idx::num_cfs]):
            for stim_idx, stim_db in enumerate(stim_dbs):

                # generate white noise stimulus
                rng = np.random.default_rng(seed=42)
                raw = rng.standard_normal(int(stim_duration * fs))
                rms = np.sqrt(np.mean(raw ** 2))
                amp = 20e-6 * 10 ** (stim_db / 20)
                waveform = amp * raw / rms

                stim = stimulus.Stimulus(waveform, fs, stim_duration)

                vihc = inner_hair_cell(
                    stimulus=stim,
                    cf=cf,
                    n_rep=1,
                    cohc=1.0,
                    cihc=1.0,
                    species=Species.CAT,
                )

                for trial in range(trials):
                    print(f"CF {cf_idx+1}/{num_cfs}, Fiber {fiber_idx+1}/{sum(num_sponts)}, Stim {stim_idx+1}/{num_stims}, Trial {trial+1}/{trials}")
                    set_seed(trial * 19)
                    ihc_mapped = map_to_synapse(
                        ihc_output=vihc,
                        spontaneous_firing_rate=fiber.spont,
                        characteristic_frequency=cf,
                        time_resolution=stim.time_resolution,
                        mapping_function=SynapseMapping.SOFTPLUS,
                    )
                    syn = synapse(
                        amplitude_ihc=ihc_mapped,
                        cf=cf,
                        n_rep=1,
                        n_timesteps=stim.n_simulation_timesteps,
                        time_resolution=stim.time_resolution,
                        spontaneous_firing_rate=fiber.spont,
                        abs_refractory_period=fiber.tabs,
                        rel_refractory_period=fiber.trel,
                        noise=NoiseType.RANDOM,
                        pla_impl=PowerLaw.APPROXIMATED,
                    )

                    psth = np.array(syn.psth)
                    Psth = np.sum(np.reshape(psth[:len(psth) // psth_bins * psth_bins], (psth_bins, -1)), axis=0)

                    tvect = np.arange(len(Psth)) * psth_bin_width
                    tstart = np.searchsorted(tvect, 1.650)
                    tend = np.searchsorted(tvect, 1.850)

                    cnt_noise[cf_idx, fiber_idx, stim_idx, trial] = np.sum(Psth[tstart:tend])

    # === Stats ===
    mean_cnt = np.mean(cnt_noise, axis=3)
    std_cnt = np.std(cnt_noise, axis=3)

    split = np.cumsum(num_sponts)
    mean_cnt_LS, mean_cnt_MS, mean_cnt_HS = np.split(mean_cnt, split[:-1], axis=1)
    std_cnt_LS, std_cnt_MS, std_cnt_HS = np.split(std_cnt, split[:-1], axis=1)

    # === Plot ===
    m = np.linspace(0, 40, 100)
    s = np.sqrt(m)

    plt.figure()
    plt.plot(mean_cnt_LS[mean_cnt_LS < 35], std_cnt_LS[mean_cnt_LS < 35], 'k^', markersize=4, label='LSR')
    plt.plot(mean_cnt_MS[mean_cnt_MS < 35], std_cnt_MS[mean_cnt_MS < 35], 'ks', markersize=4, label='MSR')
    plt.plot(mean_cnt_HS[mean_cnt_HS < 35], std_cnt_HS[mean_cnt_HS < 35], 'kx', markersize=6, label='HSR')
    plt.plot(m, s, 'k-', label=r'Poisson: $\sqrt{\mu}$')
    plt.xlim([0, 40])
    plt.ylim([0, 6])
    plt.xlabel('Mean spike count for 200ms')
    plt.ylabel('Standard deviation in spike count for 200ms')
    plt.title('Cf. Fig. 6a of Young and Barta (1986)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compute_noise_response_statistics()
