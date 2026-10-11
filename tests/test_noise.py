import os
import unittest

import librosa
import numpy as np

import brucezilany

from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

def delay(cf):
    a0 = 3.0
    a1 = 12.5
    x = 11.9 * np.log10(0.80 + (cf / 456.0))
    return a0 * np.exp(-x / a1) * 1e-3

def get_whitenoise(sampling_rate, duration):
    n_timesteps = int(np.ceil(duration * sampling_rate))
    data = np.random.normal(0, 1, n_timesteps)
    return brucezilany.stimulus.Stimulus(data, sampling_rate, duration)



class TestNoise(unittest.TestCase):
    def test_dinnoise(self):
        brucezilany.set_seed(31)

        # path = os.path.join(os.path.dirname(__file__), "tripletnoise.wav")
        # stim = brucezilany.stimulus.from_file(path, False)
        # stim.trim(0.05)

        stim = get_whitenoise(100000, 0.05)
        stim = brucezilany.stimulus.normalize_db(stim, 65.0)

        n_trials = 20
        n_fibers = 10
        
        n_low_med = int(np.floor(n_fibers / 5))
        n_high = n_fibers - (2 * n_low_med)
        
        min_freq = 150
        max_freq = 2500
        n_bins = 1
        n_mels = int(2500 / 10_500 * 64) 
        
        mel_scale = librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)

        ng = brucezilany.Neurogram(
            mel_scale,
            n_low=n_low_med,
            n_med=n_low_med,
            n_high=n_high,
            n_threads=10
        )
        ng.bin_width = 1 / (max_freq * 2 * n_bins)

        ng.create(stim, n_rep=1, n_trials=n_trials)

        data = ng.get_output().sum(axis=1)

        # breakpoint()

        data1 = savgol_filter(data[0], 13, 2)
        peaks, *_ = find_peaks(data1)

        plt.plot(data1)
        plt.scatter(peaks, data1[peaks], color='red')
        plt.show()
        # 
        # plt.figure(figsize=(10, 7))
        # plt.pcolormesh(np.arange(data.shape[1]) * ng.bin_width, mel_scale, data, cmap='gray')
        # plt.colorbar(label='# spikes per bin')
        # plt.yscale("log", base=10)
        # plt.yticks(mel_scale.astype(int), mel_scale.astype(int), fontsize=14)

        # plt.xticks(fontsize=14)
        # plt.ylabel("Freq [Hz]", fontsize=16)
        # plt.xlabel("Time [s]", fontsize=16)


        # plt.figure()
        # plt.scatter(mel_scale, delay(mel_scale))
        # plt.xlabel("CF")
        # plt.ylabel("delay")
        # plt.grid()
        # plt.show()

        
        
        
        
if __name__ == "__main__":
    unittest.main()