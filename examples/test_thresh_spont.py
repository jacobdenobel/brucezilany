import numpy as np
import matplotlib.pyplot as plt
from brucezilany import (
    synapse, 
    set_seed, 
    Species, 
    NoiseType, 
    PowerLaw, 
    SynapseMapping, 
    generate_an_population,
    find_cf_threshold,
    map_to_synapse
)




def analyze_sr_vs_threshold():
    # === Model parameters ===
    cohc = 1.0
    cihc = 1.0
    species = Species.CAT
    noise = NoiseType.RANDOM
    implnt = PowerLaw.APPROXIMATED
    mapping = SynapseMapping.SOFTPLUS

    fs = int(100e3)
    T = 1.0
    dt = 1 / fs
    nrep = 50
    time = np.arange(0, T, dt)

    numcfs = 10
    cfs = np.logspace(np.log10(400), np.log10(15e3), numcfs)
    numsponts = [4, 4, 12]
    total_fibers = sum(numsponts)

    # === Generate fibers ===
    low, med, high = generate_an_population(numcfs, *numsponts)
    fibers = low + med + high

    # === Data containers ===
    sr = np.zeros((numcfs, total_fibers))
    thresholds = np.zeros((numcfs, total_fibers))
    spontvals = np.zeros((numcfs, total_fibers))
    mean_thr_hs = np.zeros(numcfs)

    # === Silence stimulus ===
    stim = np.zeros_like(time)

    for cf_idx, cf in enumerate(cfs):
        print(f"Processing CF {cf:.1f} Hz ({cf_idx + 1}/{numcfs})")

        fibers_cf = [
            fibers[i + cf_idx * total_fibers] for i in range(total_fibers)
        ]

        for f_idx, fiber in enumerate(fibers_cf):
            set_seed(f_idx * 7)
            ihc_mapped = map_to_synapse(
                ihc_output=stim,
                spontaneous_firing_rate=fiber.spont,
                characteristic_frequency=cf,
                time_resolution=dt,
                mapping_function=mapping,
            )
            syn = synapse(
                amplitude_ihc=ihc_mapped,
                cf=cf,
                n_rep=nrep,
                n_timesteps=len(stim),
                time_resolution=dt,
                spontaneous_firing_rate=fiber.spont,
                abs_refractory_period=fiber.tabs,
                rel_refractory_period=fiber.trel,
                noise=noise,
                pla_impl=implnt,
            )

            sr[cf_idx, f_idx] = np.sum(syn.psth) / T
            spontvals[cf_idx, f_idx] = fiber.spont
            thresholds[cf_idx, f_idx] = find_cf_threshold(
                cf=cf,
                fs=fs,
                cohc=cohc,
                cihc=cihc,
                species=species,
                noise=noise,
                implnt=implnt,
                spont=fiber.spont,
                tabs=fiber.tabs,
                trel=fiber.trel,
                mapping=mapping,
            )

        # Mean threshold of fibers with SR > 18
        mask_hs = sr[cf_idx, :] > 18
        if np.any(mask_hs):
            mean_thr_hs[cf_idx] = np.mean(thresholds[cf_idx, mask_hs])

    rel_thr = thresholds - mean_thr_hs[:, None]
    mask_lm = sr <= 18

    # === Fit ===
    p = np.polyfit(np.log10(np.clip(sr[mask_lm], 0.1, None)), rel_thr[mask_lm], 1)
    p_all = np.polyfit(np.log10(np.clip(sr, 0.1, None).flatten()), rel_thr.flatten(), 1)

    # === Plots ===
    plt.figure()
    plt.semilogx(cfs / 1e3, thresholds, 'kx')
    plt.plot(cfs / 1e3, mean_thr_hs, 'r-')
    plt.ylabel('Threshold (dB SPL)')
    plt.xlabel('CF (kHz)')
    plt.xlim([0.1, 20])
    plt.xticks([0.1, 1, 10], labels=['0.1', '1', '10'])

    plt.figure()
    plt.semilogx(np.clip(sr[mask_lm], 0.1, None), rel_thr[mask_lm], 'b^')
    plt.semilogx(sr[sr > 18], rel_thr[sr > 18], 'r^')
    xvals = np.logspace(-1, 2, 100)
    plt.plot(xvals, p[0] * np.log10(xvals) + p[1], 'b-', linewidth=2.0)
    plt.text(0.15, -5, f"thrsh = {p[0]:.3f}*log10(spont)+{p[1]:.3f}")
    plt.legend(['Low & medium spont fibers', 'High spont fibers', 'Fit to LM fibers'])
    plt.xlabel('Adjusted Spont Rate (/s)')
    plt.ylabel('Relative Threshold (dB)')
    plt.xlim([0.1, 150])
    plt.xticks([0.1, 1, 10, 100], labels=['0.1', '1', '10', '100'])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    analyze_sr_vs_threshold()
