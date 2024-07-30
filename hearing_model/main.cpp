
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <chrono>

#include "bruce2018.h"


#if test_matlab
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"

bool test_random_fns(std::unique_ptr<matlab::engine::MATLABEngine>& matlabPtr, matlab::data::ArrayFactory& factory) {
	// ffGn, resample, randn, rand, sort
	std::vector<matlab::data::Array> args({
		factory.createArray<double>({ 1, 1 }, { 1 }),
		});

	matlabPtr->feval(u"rng", args);

	args = std::vector<matlab::data::Array>({
		factory.createArray<double>({ 1, 2 }, { 1, 10 }),
		});

	matlab::data::TypedArray<double> result = matlabPtr->feval(u"rand", args);

	for (auto xi : result)
		std::cout << xi << ' ';

	GENERATOR.seed(1);
	std::cout << std::endl;
	for (auto xi : rand(10)) {
		std::cout << xi << ' ';
	}
	std::cout << std::endl;
	return true;
}

std::vector<double> run_matlab_ffgn() {
	std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();
	matlab::data::ArrayFactory factory;

	std::vector<matlab::data::Array> args({
		factory.createScalar<double>(5300),
		factory.createScalar<double>(1e-4),
		factory.createScalar<double>(.9),
		factory.createScalar<int>(0),
		factory.createScalar<double>(100)
		});
	std::cout << "hallo\n";

	matlab::data::TypedArray<double> result = matlabPtr->feval(u"ffGn", args);
	std::vector<double> vResult(result.getNumberOfElements());
	std::copy(result.begin(), result.end(), vResult.begin());
	return vResult;
}
#endif

void plot(
	std::vector<std::vector<double>> vectors,
	const std::string& ptype = "line",
	const std::string& title = "title",
	const std::string& xlabel = "x",
	const std::string& ylabel = "y",
	bool detach = true
) {

	std::filesystem::path p = "C:\\Users\\Jacob\\source\\repos\\jacobdenobel\\hearing_model";
	const auto py = (p / "venv\\Scripts\\python.exe").generic_string();
	const auto plot = (p / "plot.py").generic_string();
	auto command = (detach ? "start " : "") + "py "s + plot + " " + ptype + " " + title + " " + xlabel + " " + ylabel;

	for (auto i = 0; i < vectors.size(); i++) {
		auto path = (title + std::to_string(i) + ".txt");
		std::ofstream out;
		out.open(path);

		for (auto xi : vectors[i])
			out << xi << ' ';
		out.close();
		command += " " + path;
	}

	std::cout << command << std::endl;
	system(command.c_str());
}



void test_adaptive_redocking() {

	// For adaptive redocking
	static bool make_plots = false;
	static int CF = (int)5e3;    // CF in Hz;
	static int spont = 100;      // spontaneous firing rate
	static double tabs = 0.6e-3; // Absolute refractory period
	static double trel = 0.6e-3; // Baseline mean relative refractory period
	static double cohc = 1.0;    // normal ohc function
	static double cihc = 1.0;    // normal ihc function
	static int species = 1;      // 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
	static NoiseType noiseType = FIXED_MATLAB;    // 1 for variable fGn; 0 for fixed (frozen) fGn (this is different)
	static PowerLaw implnt = APPROXIMATED;       // "0" for approximate or "1" for actual implementation of the power-law functions in the synapse (t his is reversed)
	static int nrep = 1;         // number of stimulus repetitions
	static int trials = 10000;	 // number of trails 

	// Stimulus parameters
	static double stimdb = 60.0;                // stimulus intensity in dB SPL
	static double F0 = static_cast<double>(CF); // stimulus frequency in Hz
	static int Fs = (int)100e3;                 // sampling rate in Hz (must be 100, 200 or 500 kHz)
	static double T = 0.25;                     // stimulus duration in seconds
	static double rt = 2.5e-3;                  // rise/fall time in seconds
	static double ondelay = 25e-3;				// delay for the stim


	const auto stimulus = stimulus::ramped_sine_wave(T, 2.0 * T, Fs, rt, ondelay, F0, stimdb);


	auto start = std::chrono::high_resolution_clock::now();


	auto ihc = inner_hair_cell(stimulus, CF, nrep, cohc, cihc, CAT);

	std::cout << utils::sum(stimulus.data) << std::endl;
	std::cout << utils::sum(ihc) << std::endl;

	//// This needs the new parameters
	auto pla = synapse_mapping::map(ihc, 
		spont, 
		CF, 
		stimulus.time_resolution, 
		SOFTPLUS
	);


	double psthbinwidth = 5e-4;
	size_t psthbins = (size_t)round(psthbinwidth * Fs); // number of psth bins per psth bin
	size_t n_bins = ihc.size() / psthbins;
	size_t n_bins_eb = ihc.size() / 500;
	std::vector<double> ptsh(n_bins, 0.0);

	if (make_plots)
		plot({ stimulus.data }, "line", "stimulus");

	std::vector<std::vector<double>> trd(n_bins_eb, std::vector<double>(trials));


	size_t nmax = 50;
	std::vector<std::vector<double>> synout_vectors(nmax);
	std::vector<std::vector<double>> trd_vectors(nmax, std::vector<double>(n_bins_eb));
	std::vector<std::vector<double>> trel_vectors(nmax);

	std::vector<double> n_spikes(trials);

	for (auto i = 0; i < trials; i++) {
		std::cout << i << "/" << trials << std::endl;
		auto out = synapse(pla, CF, nrep, stimulus.n_simulation_timesteps, stimulus.time_resolution, noiseType, implnt, spont, tabs, trel);
		n_spikes[i] = utils::sum(out.psth);
		auto binned = utils::make_bins(out.psth, n_bins);



		utils::scale(binned, 1.0 / trials / psthbinwidth);
		utils::add(ptsh, binned);

		for (size_t j = 0; j < n_bins_eb; j++)
			trd[j][i] = out.redocking_time[j * 500] * 1e3;

		if (i < nmax) {
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
	//assert(abs(utils::mean(ptsh) - 105.4) < 1e-8);
	//assert(ptsh[10] == 200.0);
	//assert(ptsh[15] == 200.0);

	//if (make_plots) {

	//	plot({ ptsh, t }, "bar", "PTSH", "Time(s)", "FiringRate(s)");


	//	t.resize(n_bins_eb);
	//	for (size_t i = 0; i < n_bins_eb; i++)
	//		t[i] = i * interval;

	//	auto m = utils::reduce_mean(trd);
	//	auto s = utils::reduce_std(trd, m);
	//	plot({ m, s, t }, "errorbar", "RedockingTime", "Time(s)", "t_{rd}(ms)");

	//	plot(synout_vectors, "line", "OutputRate", "x", "S_{out}");
	//	plot(trd_vectors, "line", "RelDockTime", "x", "tau_{rd}");
	//	plot(trel_vectors, "line", "RelRefr", "x", "t_{rel}");
	//}
}



std::vector<double> read_file(const std::string& fname) {
	std::ifstream f(fname);
	std::string line;
	std::vector<double> data;
	while (std::getline(f, line)) {
		std::stringstream ss(line);
		std::string number;
		while (std::getline(ss, number, ' ')) {
			data.push_back(std::stod(number));
		}
	}
	return data;
}




void example_neurogram()
{
	static double sampFreq = 10e3;
	static int CF = (int)5e3;
	static double stimdb = 60.0;                // stimulus intensity in dB SPL
	static double F0 = static_cast<double>(CF); // stimulus frequency in Hz
	static int Fs = (int)100e3;                 // sampling rate in Hz (must be 100, 200 or 500 kHz)
	static double T = 0.25;                     // stimulus duration in seconds
	static double rt = 2.5e-3;                  // rise/fall time in seconds
	static double ondelay = 25e-3;				// delay for the stim

	const double interval = 1.0 / Fs;
	const size_t mxpts = (size_t)(T / interval) + 1;
	auto pin = stimulus::ramped_sine_wave(interval, mxpts, Fs, rt, ondelay, F0, stimdb);


	auto species = HUMAN_SHERA;
	Neurogram ng(40);
	//ng.create(pin, 1, sampFreq, 1.0 / Fs, 2 * T, species, RANDOM, APPROXIMATED);
}

int main() {
	test_adaptive_redocking();
	//example_neurogram();
}