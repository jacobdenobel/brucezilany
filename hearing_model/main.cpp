
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

std::vector<double> ramped_sine_wave(const double period, const size_t n, double Fs, double rt, double ondelay, double F0, double stimdb) {
	static double pi = 3.141592653589793;
	// Generate stimulus
	const size_t irpts = (size_t)rt * (size_t)Fs;
	const size_t onbin = (size_t)std::round(ondelay * Fs); // time of first stimulus
	std::vector<double> pin(onbin + n);


	const double amplitude = std::sqrt(2.0) * 20e-6 * std::pow(10.0, (stimdb / 20.0));
	// Generate the stimulus sin wave
	for (size_t i = 0; i < n; i++)
		pin.at(onbin + i) = amplitude * std::sin(2.0 * pi * F0 * (i * period));

	// Generate the ramps
	for (size_t i = 0; i < irpts; i++)
	{
		const double ramp = static_cast<double>(i) / irpts;
		pin.at(onbin + i) = pin.at(onbin + i) * ramp;                 // upramp
		pin.at(onbin + n - i - 1) = pin.at(onbin + n - i - 1) * ramp; // downramp
	}
	return pin;
}


std::vector<double> make_bins(const std::vector<double>& x, const size_t n_bins) {
	const size_t binsize = x.size() / n_bins;
	std::vector<double> res(n_bins, 0.0);

	for (size_t i = 1; i < n_bins; i++)
		res[i] = std::accumulate(x.begin() + (i - 1) * binsize, x.begin() + i * binsize, 0.0);
	return res;
}

std::vector<double> cumsum(const std::vector<double>& x) {
	std::vector<double> res(x.size());
	res[0] = x[0];
	for (size_t i = 1; i < x.size(); i++)
		res[i] = res[i - 1] + x[i];
	return res;
}

void add(std::vector<double>& x, std::vector<double>& y) {
	for (int i = 0; i < x.size(); i++)
		x[i] += y[i];
}

void scale(std::vector<double>& x, double y) {
	for (int i = 0; i < x.size(); i++)
		x[i] *= y;
}

double variance(const std::vector<double>& x, const double m) {
	double var = 0.0;
	for (const auto& xi : x)
		var += (xi - m) * (xi - m);
	return var / x.size();

}
double stddev(const std::vector<double>& x, const double m) {
	return std::sqrt(variance(x, m));
}

double mean(const std::vector<double>& x) {
	return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}

std::vector<double> reduce_mean(const std::vector<std::vector<double>>& x) {
	std::vector<double> y(x.size());
	for (size_t i = 0; i < x.size(); i++)
		y[i] = mean(x[i]);
	return y;
}

std::vector<double> reduce_std(const std::vector<std::vector<double>>& x, const std::vector<double>& means) {
	std::vector<double> y(x.size());
	for (size_t i = 0; i < x.size(); i++)
		y[i] = stddev(x[i], means[i]);
	return y;
}




void test_adaptive_redocking() {

	// For adaptive redocking
	static bool make_plots = false;
	static double sampFreq = 10e3;
	static int CF = (int)5e3;         // CF in Hz;
	static int spont = 100;      // spontaneous firing rate
	static double tabs = 0.6e-3; // Absolute refractory period
	static double trel = 0.6e-3; // Baseline mean relative refractory period
	static double cohc = 1.0;    // normal ohc function
	static double cihc = 1.0;    // normal ihc function
	static int species = 1;      // 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
	static NoiseType noiseType = FIXED_MATLAB;    // 1 for variable fGn; 0 for fixed (frozen) fGn (this is different)
	static PowerLaw implnt = APPROXIMATED;       // "0" for approximate or "1" for actual implementation of the power-law functions in the synapse (t his is reversed)
	static int nrep = 2;         // number of stimulus repetitions
	static int trials = 10;	 // number of trails 

	// Stimulus parameters
	static double stimdb = 60.0;                // stimulus intensity in dB SPL
	static double F0 = static_cast<double>(CF); // stimulus frequency in Hz
	static int Fs = (int)100e3;                      // sampling rate in Hz (must be 100, 200 or 500 kHz)
	static double T = 0.25;                     // stimulus duration in seconds
	static double rt = 2.5e-3;                  // rise/fall time in seconds
	static double ondelay = 25e-3;				// delay for the stim

	const double interval = 1.0 / Fs;
	const size_t mxpts = (size_t)(T / interval) + 1;
	auto pin = ramped_sine_wave(interval, mxpts, Fs, rt, ondelay, F0, stimdb);

	auto start = std::chrono::high_resolution_clock::now();


	auto ihc = inner_hair_cell(pin, CF, nrep, interval, 2 * T, 1, 1, CAT);

	const int totalstim = (int)(ihc.size() / nrep);

	// This needs the new parameters
	auto pla = synapse_mapping::map(ihc,
		spont, CF, sampFreq,
		interval, SOFTPLUS);


	double psthbinwidth = 5e-4;
	size_t psthbins = (size_t)round(psthbinwidth * Fs); // number of psth bins per psth bin
	size_t n_bins = ihc.size() / psthbins;
	size_t n_bins_eb = ihc.size() / 500;
	std::vector<double> ptsh(n_bins, 0.0);

	if (make_plots)
		plot({ pin }, "line", "stimulus");

	std::vector<std::vector<double>> trd(n_bins_eb, std::vector<double>(trials));


	size_t nmax = 50;
	std::vector<std::vector<double>> synout_vectors(nmax);
	std::vector<std::vector<double>> trd_vectors(nmax, std::vector<double>(n_bins_eb));
	std::vector<std::vector<double>> trel_vectors(nmax);
	//std::cout << mean(pin) << std::endl;
	for (auto i = 0; i < trials; i++) {
		std::cout << i << "/" << trials << std::endl;
		auto out = synapse(pla, CF, nrep, totalstim, interval, noiseType, implnt, spont, tabs, trel);
		auto binned = make_bins(out.psth, n_bins);

		scale(binned, 1.0 / trials / psthbinwidth);
		add(ptsh, binned);

		for (size_t j = 0; j < n_bins_eb; j++)
			trd[j][i] = out.redocking_time[j * 500] * 1e3;

		if (i < nmax) {
			synout_vectors[i] = out.synaptic_output;
			for (size_t j = 0; j < n_bins_eb; j++)
				trd_vectors[i][j] = out.redocking_time[j * 500] * 1e3;

			trel_vectors[i] = out.mean_relative_refractory_period;
			scale(trel_vectors[i], 1e3);
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "time elapsed: " << 100.0 / duration.count() << " seconds" << std::endl;


	std::vector<double> t(n_bins);
	for (size_t i = 0; i < n_bins; i++)
		t[i] = i * psthbinwidth;

	std::cout << "expected: 101.86, actual: " << mean(ptsh) << std::endl;
	std::cout << ptsh[10] << std::endl;
	std::cout << ptsh[15] << std::endl;
	assert(abs(mean(ptsh) - 103.4) < 1e-8);
	assert(ptsh[10] == 200.0);
	assert(ptsh[15] == 200.0);

	if (make_plots) {

		plot({ ptsh, t }, "bar", "PTSH", "Time(s)", "FiringRate(s)");


		t.resize(n_bins_eb);
		for (size_t i = 0; i < n_bins_eb; i++)
			t[i] = i * interval;

		auto m = reduce_mean(trd);
		auto s = reduce_std(trd, m);
		plot({ m, s, t }, "errorbar", "RedockingTime", "Time(s)", "t_{rd}(ms)");

		plot(synout_vectors, "line", "OutputRate", "x", "S_{out}");
		plot(trd_vectors, "line", "RelDockTime", "x", "tau_{rd}");
		plot(trel_vectors, "line", "RelRefr", "x", "t_{rel}");
	}
}

template<typename T>
void print(const std::vector<T>& x) {
	for (auto& xi : x) {
		std::cout << xi << " ";
	}
	std::cout << std::endl;
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


struct Stimulus
{
	size_t frequency;
	std::vector<double> data;

	Stimulus read(const std::string path)
	{

		//[stim, Fs_stim] = audioread('defineit.wav');
		// stimdb = 65;% speech level in dB SPL
		// stim = stim / rms(stim) * 20e-6 * 10 ^ (stimdb / 20);*/
		Stimulus s;
		s = normalize_db(s);
		return s;
	}

	static Stimulus& normalize_db(Stimulus& stim)
	{
		const double stim_db = 65;

		double rms_stim = 0.0;
		for (const auto& xi : stim.data)
			rms_stim += xi * xi;
		rms_stim = std::sqrt(rms_stim / static_cast<double>(stim.data.size()));

		for (auto& xi : stim.data)
			xi = xi / rms_stim * 20e-6 * pow(10, stim_db / 20);
		return stim;
	}
};

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
	auto pin = ramped_sine_wave(interval, mxpts, Fs, rt, ondelay, F0, stimdb);


	auto species = HUMAN_SHERA;
	Neurogram ng(40);
	ng.create(pin, 1, sampFreq, 1.0 / Fs, 2 * T, species, RANDOM, APPROXIMATED);
}

int main() {
	//test_adaptive_redocking();
	example_neurogram();
}