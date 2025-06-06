#pragma once

#include "audiofile.h"
#include "utils.h"

namespace stimulus
{
	struct Stimulus
	{
		//! Data
		std::vector<double> data;
		//! sampling rate in Hz (must be 100, 200 or 500 kHz)
		size_t sampling_rate;
		//! the binsize in seconds (1 / sampling rate)
		double time_resolution;
		//! stimulus duration in seconds
		double stimulus_duration;
		//! Simulation duration, i.e. you often would like to simulate for longer than the duration of the stim
		double simulation_duration;
		//! number of stimulation timesteps (data.size)
		size_t n_stimulation_timesteps;
		//! number of simulation timesteps (rep time)
		size_t n_simulation_timesteps;

		Stimulus(
			const std::vector<double> &data,
			const size_t sampling_rate,
			const double simulation_duration) : data(data),
												sampling_rate(sampling_rate),
												time_resolution(1.0 / static_cast<double>(sampling_rate)),
												stimulus_duration(static_cast<double>(data.size()) * time_resolution),
												simulation_duration(simulation_duration),
												n_stimulation_timesteps(data.size()),
												n_simulation_timesteps(static_cast<size_t>(std::ceil(simulation_duration / time_resolution)))
		{
		}
	};

	Stimulus from_file(const std::string &path, bool verbose = true, double sim_time = 1.0, bool normalize = true);

	Stimulus ramped_sine_wave(
		double duration,
		double simulation_duration,
		size_t sampling_rate,
		double rt,
		double delay,
		double f0,
		double db);

	Stimulus normalize_db(Stimulus &stim, double stim_db = 65);
}
