import os
import unittest
from time import perf_counter

import librosa
import numpy as np

import brucezilany


class TestTiming(unittest.TestCase):
    def test_din(self):
        brucezilany.set_seed(31)
        path = os.path.join(os.path.dirname(__file__), "025.wav")
        stim = brucezilany.stimulus.from_file(path, False)

        n_trials = 1
        n_fibers = 6
        
        n_low_med = int(np.floor(n_fibers / 5))
        n_high = n_fibers - (2 * n_low_med)
        
        min_freq = 150
        max_freq = 10_500
        n_bins = 1
        n_mels = 64 
        
        mel_scale = librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)

        ng = brucezilany.Neurogram(
            mel_scale,
            n_low=n_low_med,
            n_med=n_low_med,
            n_high=n_high,
            n_threads=10
        )
        ng.bin_width = 1 / (max_freq * 2 * n_bins)

        # time = perf_counter()
        ng.create(stim, n_rep=1, n_trials=n_trials)
        # print(perf_counter() - time)
        
if __name__ == "__main__":
    unittest.main()