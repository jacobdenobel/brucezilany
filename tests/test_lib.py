import os 
import unittest

import brucezilany


class TestCase(unittest.TestCase):
    def test_types(self):
        self.assertTrue('ACTUAL' in dir(brucezilany))


    def test_stimulus(self):
        stim = brucezilany.stimulus.ramped_sine_wave(.25, .3, int(100e3), 2.5e-3, 25e-3, int(5e3), 60.0)
        self.assertAlmostEqual(stim.stimulus_duration, 0.275)
        self.assertEqual(stim.sampling_rate, int(100e3))

    def test_read_stimulus(self):
        root = os.path.dirname(os.path.dirname(__file__))
        stim = brucezilany.stimulus.from_file(os.path.join(root, "data/defineit.wav"), False)
        self.assertAlmostEqual(stim.stimulus_duration, 0.891190)
        self.assertEqual(stim.sampling_rate, int(100e3))

    def test_neurogram(self):
        stim = brucezilany.stimulus.ramped_sine_wave(
            .05, .1, int(100e3), 2.5e-3, 25e-3, int(5e3), 60.0
        )
        ng = brucezilany.Neurogram(2, 1, 1, 3)
        ng.bin_width = stim.time_resolution

        ng.create(stim, 1)

        binned_output = ng.get_output().sum(axis=1)
        self.assertEqual(binned_output.shape[0], 2)
        self.assertEqual(binned_output.shape[1], int(stim.n_simulation_timesteps / (ng.bin_width / stim.time_resolution)))
    
    
if __name__ == "__main__":
    unittest.main()