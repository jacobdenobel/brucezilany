#include "stimulus.h"
#include "resample.h"

namespace stimulus
{
	Stimulus ramped_sine_wave(
		const double duration,
		const double simulation_duration,
		const size_t sampling_rate,
		const double rt,
		const double delay,
		const double f0,
		const double db)
	{
		const double interval = 1.0 / static_cast<double>(sampling_rate);
		const size_t n = static_cast<size_t>(duration / interval) + 1;

		// Generate stimulus
		const size_t irpts = static_cast<size_t>(rt * static_cast<double>(sampling_rate));
		const size_t onbin = static_cast<size_t>(std::round(delay * static_cast<double>(sampling_rate))); // time of first stimulus

		auto stim = Stimulus{std::vector<double>(onbin + n), sampling_rate, simulation_duration};

		const double amplitude = std::sqrt(2.0) * 20e-6 * std::pow(10.0, (db / 20.0));
		// Generate the stimulus sin wave
		for (size_t i = 0; i < n; i++)
			stim.data[onbin + i] = amplitude * std::sin(2.0 * M_PI * f0 * (static_cast<double>(i) * interval));

		// Generate the ramps
		for (size_t i = 0; i < irpts; i++)
		{
			const double ramp = static_cast<double>(i) / static_cast<double>(irpts);
			stim.data[onbin + i] = stim.data[onbin + i] * ramp;					// upramp
			stim.data[onbin + n - i - 1] = stim.data[onbin + n - i - 1] * ramp; // downramp
		}
		return stim;
	}

	Stimulus normalize_db(Stimulus &stim, const double stim_db)
	{
		double rms_stim = 0.0;
		for (const auto &xi : stim.data)
			rms_stim += xi * xi;
		rms_stim = std::sqrt(rms_stim / static_cast<double>(stim.data.size()));

		for (auto &xi : stim.data)
			xi = xi / rms_stim * 20e-6 * pow(10, stim_db / 20);
		return stim;
	};

	Stimulus from_file(const std::string &path, const bool verbose, const double sim_time, const bool normalize)
	{
		if (verbose)
			std::cout << "loading file: " << path << "\n";
		const AudioFile<double> audio_file(path);
		if (verbose)
			audio_file.printSummary();

		// Assuming single channel
		assert(audio_file.samples.size() == 1);

		constexpr int required_sample_rate = 100e3;
		const auto sample_rate = audio_file.getSampleRate();
		std::vector<double> data = audio_file.samples[0];
		data = resample(required_sample_rate, sample_rate, data);

		const auto stim_duration = static_cast<double>(data.size()) * 1.0 / required_sample_rate;
		auto stim = Stimulus(data, required_sample_rate, sim_time * stim_duration);
		
		if (sim_time == 1.0) // fix precision errors
			stim.simulation_duration = stim.stimulus_duration;

		if (!normalize)
			return stim;

		return normalize_db(stim);
	}
}
