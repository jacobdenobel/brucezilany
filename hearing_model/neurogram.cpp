#include "neurogram.h"


Neurogram::Neurogram(const size_t n_cf) :
	cfs_(utils::log_space(std::log10(250.0), std::log10(16e3), n_cf)),
	// assumes no hearing loss
	db_loss_(n_cf, 0.0),
	coh_cs_(n_cf, 1.0),
	ihc_cs_(n_cf, 1.0),
	ohc_loss_(n_cf, 0.0),
	an_population_(generate_an_population(n_cf, 10, 10, 30))
{
}

std::vector<Fiber> Neurogram::generate_fiber_set(
	const size_t n_cf,
	const size_t n_fibers,
	const FiberType f_type,
	const double c1,
	const double c2,
	const double amin,
	const double amax
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
			std::min(std::max(c1 + c2 * utils::randn1(), amin), amax),
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
	const size_t cf_i,
	const std::vector<double>& ihc,
	const int n_rep,
	const double sampling_freq,
	const double time_resolution,
	const NoiseType noise_type,
	const PowerLaw power_law,
	const size_t n_timesteps,
	const std::vector<double>& smw_ft,
	const Fiber& fiber
)
{
	/*const auto pla = synapse_mapping::map(ihc,
	                                      fiber.spont,
	                                      cfs_[cf_i],
	                                      sampling_freq,
	                                      time_resolution,
	                                      SOFTPLUS
	);
	const auto out = synapse(pla,
	                         cfs_[cf_i], n_rep,
	                         static_cast<int>(n_timesteps),
	                         time_resolution,
	                         noise_type,
	                         power_law,
	                         fiber.spont,
	                         fiber.tabs,
	                         fiber.trel
	);
	mutex_.lock();
	utils::add(fine_timing_[cf_i], utils::filter(smw_ft, out.psth));
	mutex_.unlock();*/
}


void Neurogram::evaluate_cf(
	const size_t cf_i,
	const std::vector<double>& sound_wave,
	const int n_rep,
	const double sampling_freq,
	const double time_resolution,
	const double rep_time,
	const Species species,
	const NoiseType noise_type,
	const PowerLaw power_law,
	const size_t n_timesteps,
	const std::vector<double>& smw_ft
)
{
	const auto fibers = get_fibers(cf_i);

	/*const auto ihc = inner_hair_cell(
		sound_wave, cfs_[cf_i], n_rep, time_resolution, rep_time, coh_cs_[cf_i], ihc_cs_[cf_i], species
	);

	assert(static_cast<int>(ihc.size() / n_rep) == n_timesteps);

	std::vector<std::thread> threads(fibers.size());
	for (size_t f_id = 0; f_id < fibers.size(); f_id++)
	{
		threads[f_id] = std::thread(&Neurogram::evaluate_fiber, this, cf_i, ihc, n_rep, sampling_freq, time_resolution,
			noise_type, power_law, n_timesteps, smw_ft, fibers[f_id]);
	}

	for (auto& th : threads)
		th.join();*/

	std::cout << cf_i << '\n';
}


void Neurogram::create(
	const std::vector<double>& sound_wave,
	const int n_rep,
	const double sampling_freq,
	const double time_resolution,
	const double rep_time,
	const Species species,
	const NoiseType noise_type,
	const PowerLaw power_law
)
{
	const auto smw_ft = utils::hamming(32);
	const auto smw_mr = utils::hamming(128);

	//constexpr double psthbinwidth_mr = 100e-6;
	//const double psthbins = round(psthbinwidth_mr * Fs);

	const auto n_timesteps = static_cast<int>(std::ceil(rep_time / time_resolution));
	fine_timing_ = std::vector(cfs_.size(), std::vector(n_timesteps, 0.0));


	std::vector<std::thread> threads(cfs_.size());
	for (size_t cf_i = 0; cf_i < cfs_.size(); cf_i++)
	{
		threads[cf_i] = std::thread(&Neurogram::evaluate_cf, this, cf_i, sound_wave, n_rep, sampling_freq,
		                            time_resolution, rep_time, species, noise_type, power_law, n_timesteps, smw_ft);
	}

	for (auto& th : threads)
		th.join();

	std::cout << "done\n";
}
