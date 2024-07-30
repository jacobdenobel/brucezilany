#include "stimulus.h"


namespace stimulus
{
	Stimulus ramped_sine_wave(
		const double duration,
		const double simulation_duration,
		const size_t sampling_rate,
		const double rt,
		const double delay,
		const double f0,
		const double db
	)
	{
		const double interval = 1.0 / static_cast<double>(sampling_rate);
		const size_t n = static_cast<size_t>(duration / interval) + 1;

		// Generate stimulus
		const size_t irpts = static_cast<size_t>(rt * static_cast<double>(sampling_rate));
		const size_t onbin = static_cast<size_t>(std::round(delay * static_cast<double>(sampling_rate))); // time of first stimulus

		auto stim = Stimulus{ std::vector<double>(onbin + n), sampling_rate, simulation_duration};

		const double amplitude = std::sqrt(2.0) * 20e-6 * std::pow(10.0, (db / 20.0));
		// Generate the stimulus sin wave
		for (size_t i = 0; i < n; i++)
			stim.data[onbin + i] = amplitude * std::sin(2.0 * M_PI * f0 * (static_cast<double>(i) * interval));

		// Generate the ramps
		for (size_t i = 0; i < irpts; i++)
		{
			const double ramp = static_cast<double>(i) / static_cast<double>(irpts);
			stim.data[onbin + i] = stim.data[onbin + i] * ramp; // upramp
			stim.data[onbin + n - i - 1] = stim.data[onbin + n - i - 1] * ramp; // downramp
		}
		return stim;
	}

	Stimulus& normalize_db(Stimulus& stim)
	{
		constexpr double stim_db = 65;

		double rms_stim = 0.0;
		for (const auto& xi : stim.data)
			rms_stim += xi * xi;
		rms_stim = std::sqrt(rms_stim / static_cast<double>(stim.data.size()));

		for (auto& xi : stim.data)
			xi = xi / rms_stim * 20e-6 * pow(10, stim_db / 20);
		return stim;
	};


	//Stimulus read_stimulus(const std::string path)
	//{
	//	//[stim, Fs_stim] = audioread('defineit.wav');
	//	// stimdb = 65;% speech level in dB SPL
	//	// stim = stim / rms(stim) * 20e-6 * 10 ^ (stimdb / 20);*/
	//	/*Stimulus s;
	//	s = normalize_db(s);
	//	return s;*/
	//};
}
