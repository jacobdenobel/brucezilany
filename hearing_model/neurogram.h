#pragma once

#include <array>
#include <mutex>
#include "utils.h"
#include "inner_hair_cell.h"


enum FiberType
{
	LOW = 0, MEDIUM = 1, HIGH = 2
};

struct Fiber
{
	double spont;
	double tabs;
	double trel;
	FiberType type;
};


class Neurogram
{
	std::vector<double> cfs_;
	std::vector<double> db_loss_;
	std::vector<double> coh_cs_;
	std::vector<double> ihc_cs_;
	std::vector<double> ohc_loss_;

	std::array<std::vector<Fiber>, 3> an_population_;

	std::vector<std::vector<double>> fine_timing_;
	std::vector<std::vector<double>> mean_timing_;
	std::mutex mutex_;

	std::vector<double> hamming_window_ft_;
	std::vector<double> hamming_window_mr_;

public:
	explicit Neurogram(size_t n_cf = 40);

	static std::vector<Fiber> generate_fiber_set(size_t n_cf, size_t n_fibers, FiberType f_type, double c1, double c2, double lb, double ub);

	static std::array<std::vector<Fiber>, 3> generate_an_population(
		size_t n_cf,
		size_t n_low,
		size_t n_med,
		size_t n_high);


	[[nodiscard]] std::vector<Fiber> get_fibers(size_t cf_idx) const;

	void create(
		const stimulus::Stimulus& sound_wave,
		int n_rep,
		Species species,
		NoiseType noise_type,
		PowerLaw power_law
	);

	void evaluate_cf(
		const stimulus::Stimulus& sound_wave,
		int n_rep,
		Species species,
		NoiseType noise_type,
		PowerLaw power_law,
		size_t cf_i
	);

	void evaluate_fiber(
		const stimulus::Stimulus& sound_wave,
		const std::vector<double>& ihc,
		int n_rep,
		NoiseType noise_type,
		PowerLaw power_law,
		const Fiber& fiber,
		size_t cf_i
	);

	[[nodiscard]] std::vector<std::vector<double>> get_fine_timing() const 
	{
		return fine_timing_;
	}

	[[nodiscard]] std::vector<std::vector<double>> get_mean_timing() const
	{
		return mean_timing_;
	}

	[[nodiscard]] std::vector<double> get_cfs() const
	{
		return cfs_;
	}

};

