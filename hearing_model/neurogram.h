#pragma once

#include <array>
#include "utils.h"
#include "inner_hair_cell.h"
#include "synapse.h"

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

public:
	explicit Neurogram(size_t n_cf = 40);

	static std::vector<Fiber> generate_fiber_set(size_t n_cf, size_t n_fibers,  FiberType f_type, double c1, double c2, double amin, double amax);

	static std::array<std::vector<Fiber>, 3> generate_an_population(
		size_t n_cf,
		size_t n_low,
		size_t n_med,
		size_t n_high);


	[[nodiscard]] std::vector<Fiber> get_fibers(size_t cf_idx) const;

	void create(
		const std::vector<double>& sound_wave,
		int n_rep,
		double sampling_freq, 
		double time_resolution,
		double rep_time,
		Species species,
		NoiseType noise_type,
		PowerLaw power_law
	);

};

