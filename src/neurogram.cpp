#include "neurogram.h"

#include <cassert>
#include <thread>
#include <future>
#include "synapse.h"
#include "synapse_mapping.h"

static int get_n_threads(const int n_jobs)
{
	const int processor_count = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    const int n_threads = n_jobs == -1 ? processor_count : std::min(std::max(1, n_jobs), processor_count);
	return n_threads;
}

Neurogram::Neurogram(
	const std::vector<double> &cfs,
	const size_t n_low,
	const size_t n_med,
	const size_t n_high,
	const int n_threads
) : cfs_(cfs),
						   // assumes no hearing loss
						   db_loss_(cfs.size(), 0.0),
						   coh_cs_(cfs.size(), 1.0),
						   ihc_cs_(cfs.size(), 1.0),
						   ohc_loss_(cfs.size(), 0.0),
						   an_population_(generate_an_population(cfs.size(), n_low, n_med, n_high)),
						   pool_(get_n_threads(n_threads))
{
}

Neurogram::Neurogram(
	const size_t n_cf,
	const size_t n_low,
	const size_t n_med,
	const size_t n_high,
	const int n_threads
) : Neurogram(utils::log_space(std::log10(250.0), std::log10(16e3), n_cf), n_low, n_med, n_high, n_threads)
{
}

std::vector<Fiber> Neurogram::generate_fiber_set(
	const size_t n_cf,
	const size_t n_fibers,
	const FiberType f_type,
	const double c1,
	const double c2,
	const double lb,
	const double ub,
	utils::RandomGenerator& rng
	)
{
	static double tabs_max = 1.5 * 461e-6;
	static double tabs_min = 1.5 * 139e-6;
	static double trel_max = 894e-6;
	static double trel_min = 131e-6;

	auto fibers = std::vector<Fiber>(n_cf * n_fibers);
	for (auto &fiber : fibers)
	{
		const double ref_rand = rng.rand1();
		fiber = Fiber{
			std::min(std::max(c1 + c2 * rng.randn1(), lb), ub),
			(tabs_max - tabs_min) * ref_rand + tabs_min,
			(trel_max - trel_min) * ref_rand + trel_min,
			f_type};
	}

	return fibers;
}

std::array<std::vector<Fiber>, 3> Neurogram::generate_an_population(
	const size_t n_cf,
	const size_t n_low,
	const size_t n_med,
	const size_t n_high)
{
	auto rng = utils::RandomGenerator(utils::SEED);
	return {
		generate_fiber_set(n_cf, n_low, LOW, .1, .1, 1e-3, .2, rng),
		generate_fiber_set(n_cf, n_med, MEDIUM, 4.0, 4.0, .2, 18.0, rng),
		generate_fiber_set(n_cf, n_high, HIGH, 70.0, 30, 18.0, 180.,rng)
	};
}

std::vector<Fiber> Neurogram::get_fibers(const size_t cf_idx) const
{
	std::vector<Fiber> fibers;
	for (const auto &fiber_set : an_population_)
	{
		const auto n_fibers = fiber_set.size() / cfs_.size();
		for (size_t i = cf_idx * n_fibers; i < (cf_idx + 1) * n_fibers; i++)
			fibers.push_back(fiber_set[i]);
	}
	return fibers;
}

void Neurogram::evaluate_fiber(
	const stimulus::Stimulus &sound_wave,
	const std::vector<double> &ihc,
	const int n_rep,
	const int n_trials,
	const NoiseType noise_type,
	const PowerLaw power_law,
	const Fiber &fiber,
	const size_t cf_i)
{
	const auto pla = synapse_mapping::map(
		ihc,
		fiber.spont,
		cfs_[cf_i],
		sound_wave.time_resolution,
		SOFTPLUS);

	for(int i = 0; i < n_trials; i++) {
		auto rng = utils::RandomGenerator(utils::SEED + (1 + cf_i) * (1 + i) * 1500);
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
			fiber.trel,
			false,
			rng
			);

		
		auto output = utils::make_bins(out.psth, output_[0][0].size());
		mutex_.lock();
 		utils::add(output_[cf_i][i], output);
		mutex_.unlock();
	}

}

void Neurogram::evaluate_cf(
	const stimulus::Stimulus &sound_wave,
	const int n_rep,
	const int n_trials,
	const Species species,
	const NoiseType noise_type,
	const PowerLaw power_law,
	const size_t cf_i)
{
	const auto fibers = get_fibers(cf_i);

	const auto ihc = inner_hair_cell(
		sound_wave, cfs_[cf_i], n_rep, coh_cs_[cf_i], ihc_cs_[cf_i], species
	);

	assert(ihc.size() / n_rep == sound_wave.n_simulation_timesteps);

	auto evaluate_fiber_ = [this, n_rep, n_trials, noise_type, power_law, cf_i](
		int idx, const Fiber& fiber, const std::vector<double>& ihc, const stimulus::Stimulus &sound_wave
	){
		return this->evaluate_fiber(sound_wave, ihc, n_rep, n_trials, noise_type, power_law, fiber, cf_i);
	};

	for (size_t f_id = 0; f_id < fibers.size(); f_id++)
		pool_.push(evaluate_fiber_, fibers[f_id], ihc, sound_wave);
}

void Neurogram::create(
	const stimulus::Stimulus &sound_wave,
	const int n_rep,
	const int n_trials,
	const Species species,
	const NoiseType noise_type,
	const PowerLaw power_law)
{
	bin_width = std::max(bin_width, sound_wave.time_resolution);
	const double time_ratio = bin_width / sound_wave.time_resolution;
	const double n_bins_ratio = static_cast<double>(sound_wave.n_simulation_timesteps) / time_ratio;
	const size_t n_bins = static_cast<size_t>(std::round(n_bins_ratio));
	output_ = std::vector(cfs_.size(), std::vector(n_trials, std::vector(n_bins, 0.0)));

	pool_.resize(std::max(static_cast<int>(pool_.size()) - static_cast<int>(cfs_.size()), static_cast<int>(cfs_.size())));
	
	std::vector<std::thread> threads(cfs_.size());

	for (size_t cf_i = 0; cf_i < cfs_.size(); cf_i++)
		threads[cf_i] = std::thread(
			&Neurogram::evaluate_cf, this, sound_wave, n_rep, n_trials, species, noise_type, power_law, cf_i);

	for (auto &th : threads)
		th.join();

	pool_.stop(true);
}
