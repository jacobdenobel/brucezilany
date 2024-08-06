#include "neurogram.h"

#include <cassert>
#include <thread>
#include "synapse.h"
#include "synapse_mapping.h"


Neurogram::Neurogram(const size_t n_cf) :
	cfs_(utils::log_space(std::log10(250.0), std::log10(16e3), n_cf)),
	// assumes no hearing loss
	db_loss_(n_cf, 0.0),
	coh_cs_(n_cf, 1.0),
	ihc_cs_(n_cf, 1.0),
	ohc_loss_(n_cf, 0.0),
	an_population_(generate_an_population(n_cf, 10, 10, 30)),
	hamming_window_ft_(utils::hamming(32)),
	hamming_window_mr_(utils::hamming(128))
{
}

std::vector<Fiber> Neurogram::generate_fiber_set(
	const size_t n_cf,
	const size_t n_fibers,
	const FiberType f_type,
	const double c1,
	const double c2,
	const double lb,
	const double ub
)
{
	static double tabs_max = 1.5 * 461e-6;
	static double tabs_min = 1.5 * 139e-6;
	static double trel_max = 894e-6;
	static double trel_min = 131e-6;

	auto fibers = std::vector<Fiber>(n_cf * n_fibers);
	for (auto& fiber : fibers)
	{
		const double ref_rand = utils::rand1();
		fiber = Fiber{
			std::min(std::max(c1 + c2 * utils::randn1(), lb), ub),
			(tabs_max - tabs_min) * ref_rand + tabs_min,
			(trel_max - trel_min) * ref_rand + trel_min,
			f_type
		};
	}

	return fibers;
}


std::array<std::vector<Fiber>, 3> Neurogram::generate_an_population(
	const size_t n_cf,
	const size_t n_low,
	const size_t n_med,
	const size_t n_high
)
{
	return {
		generate_fiber_set(n_cf, n_low, LOW, .1, .1, 1e-3, .2),
		generate_fiber_set(n_cf, n_med, MEDIUM, 4.0, 4.0, .2, 18.0),
		generate_fiber_set(n_cf, n_high, HIGH, 70.0, 30, 18.0, 180.)
	};
}


std::vector<Fiber> Neurogram::get_fibers(const size_t cf_idx) const
{
	std::vector<Fiber> fibers;
	for (const auto& fiber_set : an_population_)
	{
		const auto n_fibers = fiber_set.size() / cfs_.size();
		for (size_t i = cf_idx * n_fibers; i < (cf_idx + 1) * n_fibers; i++)
			fibers.push_back(fiber_set[i]);
	}
	return fibers;
}


void Neurogram::evaluate_fiber(
	const stimulus::Stimulus& sound_wave,
	const std::vector<double>& ihc,
	const int n_rep,
	const NoiseType noise_type,
	const PowerLaw power_law,
	const Fiber& fiber,
	const size_t cf_i
)
{
	const auto pla = synapse_mapping::map(
		ihc,
		fiber.spont,
		cfs_[cf_i],
		sound_wave.time_resolution,
		SOFTPLUS
	);

	const auto out = synapse(
		pla,
		cfs_[cf_i],
		n_rep,
		sound_wave.n_simulation_timesteps,
		sound_wave.time_resolution,
		noise_type,
		power_law,
		fiber.spont,
		fiber.tabs,
		fiber.trel
	);

	const auto filtered_output = utils::filter(hamming_window_ft_, out.psth);
	const auto mr_filtered_output = utils::filter(hamming_window_mr_, utils::make_bins(out.psth, mean_timing_[0].size()));

	mutex_.lock();
	utils::add(fine_timing_[cf_i], filtered_output);
	utils::add(mean_timing_[cf_i], mr_filtered_output);
	mutex_.unlock();
}


void Neurogram::evaluate_cf(
	const stimulus::Stimulus& sound_wave,
	const int n_rep,
	const Species species,
	const NoiseType noise_type,
	const PowerLaw power_law,
	const size_t cf_i
)
{
	const auto fibers = get_fibers(cf_i);

	const auto ihc = inner_hair_cell(
		sound_wave, cfs_[cf_i], n_rep, coh_cs_[cf_i], ihc_cs_[cf_i], species
	);

	assert(ihc.size() / n_rep == sound_wave.n_simulation_timesteps);

	std::vector<std::thread> threads(fibers.size());
	for (size_t f_id = 0; f_id < fibers.size(); f_id++)
		threads[f_id] = std::thread(
			&Neurogram::evaluate_fiber, this, sound_wave, ihc, n_rep, noise_type, power_law, fibers[f_id], cf_i
		);

	for (auto& th : threads)
		th.join();
}


void Neurogram::create(
	const stimulus::Stimulus& sound_wave,
	const int n_rep,
	const Species species,
	const NoiseType noise_type,
	const PowerLaw power_law
)
{
	constexpr double mr_bin_width = 100e-6;
	const size_t n_bins = sound_wave.n_simulation_timesteps / static_cast<size_t>(std::round(mr_bin_width / sound_wave.time_resolution));

	fine_timing_ = std::vector(cfs_.size(), std::vector(sound_wave.n_simulation_timesteps, 0.0));
	mean_timing_ = std::vector(cfs_.size(), std::vector(n_bins, 0.0));
	dt_fine_timing_ = static_cast<double>(hamming_window_ft_.size()) / 2.0 * sound_wave.time_resolution;
	dt_mean_timing_ = static_cast<double>(hamming_window_mr_.size()) / 2.0 * sound_wave.time_resolution * mr_bin_width / sound_wave.time_resolution;

	std::vector<std::thread> threads(cfs_.size());
	for (size_t cf_i = 0; cf_i < cfs_.size(); cf_i++)
		threads[cf_i] = std::thread(
			&Neurogram::evaluate_cf, this, sound_wave, n_rep, species, noise_type, power_law, cf_i
		);

	for (auto& th : threads)
		th.join();

	// Resample
	for (auto& xc : fine_timing_)
		xc = utils::subsequence(xc, 0, xc.size(), hamming_window_ft_.size() / 2);

	for (auto& xc : mean_timing_)
		xc = utils::subsequence(xc, 0, xc.size(), hamming_window_mr_.size() / 2);

}

