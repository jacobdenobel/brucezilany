#pragma once

#include "utils.h"

namespace stimulus
{
	struct Stimulus
	{
		size_t frequency;
		std::vector<double> data;
		double duration;
	};

	std::vector<double> ramped_sine_wave(double period, size_t n, double fs, double rt, double delay, double f0,
	                                     double db);

	Stimulus& normalize_db(Stimulus& stim);

	Stimulus read_stimulus(std::string path);
}
