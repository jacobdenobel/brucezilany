/*
 * Questions to Bruce:
 *	1. Power law mapping, should this be k-1 for post stimulus powerLawIn?
 */

#include <iostream>
#include <fstream>
#include <filesystem>

#include <chrono>

#include "bruce.h"

void test_adaptive_redocking()
{
	// For adaptive redocking
	static bool make_plots = true;
	static int CF = (int)5e3;			   // CF in Hz;
	static int spont = 100;				   // spontaneous firing rate
	static double tabs = 0.6e-3;		   // Absolute refractory period
	static double trel = 0.6e-3;		   // Baseline mean relative refractory period
	static double cohc = 1.0;			   // normal ohc function
	static double cihc = 1.0;			   // normal ihc function
	static Species species = CAT;		   // 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
	static NoiseType noiseType = RANDOM;   // 1 for variable fGn; 0 for fixed (frozen) fGn (this is different)
	static PowerLaw implnt = APPROXIMATED; // "0" for approximate or "1" for actual implementation of the power-law functions in the synapse (t his is reversed)
	static int nrep = 1;				   // number of stimulus repetitions
	static int trials = 1000;			   // number of trails

	// Stimulus parameters
	static double stimdb = 60.0;				// stimulus intensity in dB SPL
	static double F0 = static_cast<double>(CF); // stimulus frequency in Hz
	static int Fs = (int)100e3;					// sampling rate in Hz (must be 100, 200 or 500 kHz)
	static double T = 0.25;						// stimulus duration in seconds
	static double rt = 2.5e-3;					// rise/fall time in seconds
	static double ondelay = 25e-3;				// delay for the stim

	const auto stimulus = stimulus::ramped_sine_wave(T, 2.0 * T, Fs, rt, ondelay, F0, stimdb);

	// if (make_plots)
	// plot({ stimulus.data }, "line", "stimulus");

	auto start = std::chrono::high_resolution_clock::now();

	auto ihc = inner_hair_cell(stimulus, CF, nrep, cohc, cihc, species);

	std::cout << utils::sum(stimulus.data) << std::endl;
	std::cout << utils::sum(ihc) << std::endl;

	//// This needs the new parameters
	auto pla = synapse_mapping::map(ihc,
									spont,
									CF,
									stimulus.time_resolution,
									SOFTPLUS);

	if (make_plots)
		utils::plot({ihc}, "line", "ihc");

	if (make_plots)
		utils::plot({pla}, "line", "pla");

	double psthbinwidth = 5e-4;
	size_t psthbins = static_cast<size_t>(round(psthbinwidth * Fs)); // number of psth bins per psth bin
	size_t n_bins = ihc.size() / psthbins;
	size_t n_bins_eb = ihc.size() / 500;
	std::vector<double> ptsh(n_bins, 0.0);

	std::vector<std::vector<double>> trd(n_bins_eb, std::vector<double>(trials));

	size_t nmax = 50;
	std::vector<std::vector<double>> synout_vectors(nmax);
	std::vector<std::vector<double>> trd_vectors(nmax, std::vector<double>(n_bins_eb));
	std::vector<std::vector<double>> trel_vectors(nmax);

	std::vector<double> n_spikes(trials);

	for (auto i = 0; i < trials; i++)
	{
		std::cout << i << "/" << trials << '\n';
		auto out = synapse(pla, CF, nrep, stimulus.n_simulation_timesteps, stimulus.time_resolution, noiseType, implnt, spont, tabs, trel);

		n_spikes[i] = utils::sum(out.psth);
		auto binned = utils::make_bins(out.psth, n_bins);

		utils::scale(binned, 1.0 / trials / psthbinwidth);
		utils::add(ptsh, binned);

		for (size_t j = 0; j < n_bins_eb; j++)
			trd[j][i] = out.redocking_time[j * 500] * 1e3;

		if (i < nmax)
		{
			synout_vectors[i] = out.synaptic_output;
			for (size_t j = 0; j < n_bins_eb; j++)
				trd_vectors[i][j] = out.redocking_time[j * 500] * 1e3;

			trel_vectors[i] = out.mean_relative_refractory_period;
			utils::scale(trel_vectors[i], 1e3);
		}
	}

	std::cout << utils::mean(n_spikes) << '\n';
	std::cout << utils::std(n_spikes, utils::mean(n_spikes)) << '\n';

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "time elapsed: " << 100.0 / duration.count() << " seconds" << std::endl;

	std::vector<double> t(n_bins);
	for (size_t i = 0; i < n_bins; i++)
		t[i] = i * psthbinwidth;

	std::cout << "expected: 101.86, actual: " << utils::mean(ptsh) << std::endl;
	std::cout << ptsh[10] << std::endl;
	std::cout << ptsh[15] << std::endl;
	// assert(abs(utils::mean(ptsh) - 105.4) < 1e-8);
	// assert(ptsh[10] == 200.0);
	// assert(ptsh[15] == 200.0);

	if (make_plots)
	{
		utils::plot({ptsh, t}, "bar", "PTSH", "Time[s]", "FiringRate[s]");

		t.resize(n_bins_eb);
		for (size_t i = 0; i < n_bins_eb; i++)
			t[i] = i / Fs;

		auto m = utils::reduce_mean(trd);
		auto s = utils::reduce_std(trd, m);
		utils::plot({m, s, t}, "errorbar", "RedockingTime", "Time(s)", "t_{rd}(ms)");
		utils::plot(synout_vectors, "line", "OutputRate", "x", "S_{out}");
		utils::plot(trd_vectors, "line", "RelDockTime", "x", "tau_{rd}");
		utils::plot(trel_vectors, "line", "RelRefr", "x", "t_{rel}");
	}
}

void plot_neurogram(const stimulus::Stimulus &stim)
{
	Neurogram ng(40);
	ng.create(stim, 1, HUMAN_SHERA, RANDOM, APPROXIMATED);
	auto [plot_data, dt_fine_timing] = ng.get_fine_timing();
	plot_data.push_back(ng.get_cfs());
	utils::plot(plot_data, "colormesh", "fine_timing", "time", "frequency", std::to_string(dt_fine_timing));

	auto [plot_data2, dt_mean_timing] = ng.get_mean_timing();
	plot_data2.push_back(ng.get_cfs());
	utils::plot(plot_data2, "colormesh", "mean_timing", "time", "frequency", std::to_string(dt_mean_timing));
}

void example_neurogram_sin()
{
	constexpr static int cf = 5e3;			// Characteristic frequency
	constexpr static double stim_db = 60.0; // stimulus intensity in dB SPL
	constexpr static double f0 = cf;		// stimulus frequency in Hz
	constexpr static int fs = 100e3;		// sampling rate in Hz (must be 100, 200 or 500 kHz)
	constexpr static double T = 0.25;		// stimulus duration in seconds
	constexpr static double rt = 2.5e-3;	// rise/fall time in seconds
	constexpr static double delay = 25e-3;	// delay for the stim

	const auto stim = stimulus::ramped_sine_wave(T, 1.2 * (T + delay), fs, rt, delay, f0, stim_db);
	plot_neurogram(stim);
}

void example_neurogram()
{
	std::filesystem::path root = PROJECT_ROOT;
	const auto stim = stimulus::from_file(root / "data" / "defineit.wav", true);
	plot_neurogram(stim);
}

int main(int argc, char **argv)
{
	const std::string selection = (argc > 1) ? argv[1] : "neurogram_sin";

	if (selection == "redock")
		test_adaptive_redocking();
	else if (selection == "neurogram")
		example_neurogram();
	else if (selection == "neurogram_sin")
		example_neurogram_sin();
}