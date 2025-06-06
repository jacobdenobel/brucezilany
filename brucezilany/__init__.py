import numpy as np

from .brucezilanycpp import *

BARK_SCALE = np.array([
    50,
    100,
    150,
    250,
    350,
    450,
    570,
    700,
    840,
    1000,
    1170,
    1370,
    1600,
    1850,
    2150,
    2500,
    2900,
    3400,
    4000,
    4800,
    5800,
    7000,
    8500,
    10500,
    13500,
])


def find_cf_threshold(
    cf,
    fs,
    cohc,
    cihc,
    species,
    noise,
    implnt,
    spont,
    tabs,
    trel,
    mapping,
    f0=None,
    threshold_rate=5.0,
    min_db=-10,
    max_db=100,
    step_db=1.0,
    duration=0.05,
    rt=2.5e-3,
    nrep=10,
):
    """
    Find the minimum dB SPL at which firing rate exceeds `threshold_rate`.

    Returns:
        float: threshold in dB SPL
    """

    if f0 is None:
        f0 = cf

    db_range = np.arange(min_db, max_db + step_db, step_db)

    for db in db_range:
        stim = stimulus.ramped_sine_wave(
            duration=duration,
            simulation_duration=2 * duration,
            sampling_rate=fs,
            rt=rt,
            delay=0,
            f0=f0,
            db=db,
        )

        ihc = inner_hair_cell(
            stimulus=stim,
            cf=cf,
            n_rep=nrep,
            cohc=cohc,
            cihc=cihc,
            species=species,
        )

        ihc_mapped = map_to_synapse(
            ihc_output=ihc,
            spontaneous_firing_rate=spont,
            characteristic_frequency=cf,
            time_resolution=stim.time_resolution,
            mapping_function=mapping,
        )

        syn_out = synapse(
            amplitude_ihc=ihc_mapped,
            cf=cf,
            n_rep=nrep,
            n_timesteps=stim.n_simulation_timesteps,
            time_resolution=stim.time_resolution,
            spontaneous_firing_rate=spont,
            abs_refractory_period=tabs,
            rel_refractory_period=trel,
            noise=noise,
            pla_impl=implnt,
        )

        psth = np.array(syn_out.psth)
        ronset = int(round(1.5e-3 / stim.time_resolution))
        roffset = int(round(duration / stim.time_resolution))

        if ronset + roffset > len(psth):
            continue

        fr = np.sum(psth[ronset:roffset]) / (roffset * stim.time_resolution * nrep)
        if fr >= threshold_rate:
            return db

    return max_db + 10  # no threshold found within range