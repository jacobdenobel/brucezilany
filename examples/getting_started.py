import os
import numpy as np
from brucezilany import inner_hair_cell, synapse, stimulus, map_to_synapse, Neurogram, Species



def mean_firing_rate():
    stim = stimulus.ramped_sine_wave(
        duration=0.05,
        simulation_duration=0.1,
        sampling_rate=100000,
        rt=0.005,
        delay=0.01,
        f0=1000,
        db=65
    )

    # IHC response
    ihc_output = inner_hair_cell(stim, cf=1000, n_rep=10)

    # Intermediate mapper (Required!)
    mapped_ihc = map_to_synapse(
        ihc_output=ihc_output,
        spontaneous_firing_rate=100,
        characteristic_frequency=1000,
        time_resolution=stim.time_resolution,
    )

    # Synapse response
    out = synapse(
        amplitude_ihc=mapped_ihc,
        cf=1000,
        n_rep=10,
        n_timesteps=stim.n_simulation_timesteps,
        spontaneous_firing_rate=100
    )

    print("Mean firing rate:", np.mean(out.mean_firing_rate))

def stim_from_file():
    root = os.path.dirname(os.path.dirname(__file__))
    stim = stimulus.from_file(f"{root}/data/defineit.wav", verbose=False)
    stimulus.normalize_db(stim, stim_db=70)
    print("Loaded stim", stim)

    ng = Neurogram(n_cf=50, n_low=5, n_med=5, n_high=10)
    ng.create(sound_wave=stim, species=Species.HUMAN_SHERA, n_trials=3)

    output = ng.get_output() 
    print("Generated Neurogram, shape:", output.shape)



def main():
    # mean_firing_rate()
    stim_from_file()

if __name__ == "__main__":
    main()