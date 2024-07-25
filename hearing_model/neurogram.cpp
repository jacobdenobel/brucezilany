#include "neurogram.h"

#include <corecrt_math_defines.h>

#include "synapse_mapping.h"

/**
 * Generate a log_space, works similar to how np.log_space or matlabs log_space works.
 *
 * @param start start of the range
 * @param end end of the range
 * @param n the number of points to be generated
 * @return a vector of n log10 space points
 */
static std::vector<double> log_space(const double start, const double end, const size_t n)
{
	std::vector<double> space(n);
	const double step = (end - start) / (static_cast<double>(n) - 1.0);
	double current = start;
	for (size_t i = 0; i < n; i++)
	{
		space[i] = pow(10.0, current);
		current += step;
	}
	return space;
}

static std::vector<double> hamming(const size_t n)
{
	std::vector<double> window(n);
	for (size_t i = 0; i < n; i++)
		window[i] = 0.54 - 0.46 * std::cos(2 * M_PI * static_cast<double>(i) / (static_cast<double>(n) - 1));
	return window;
}

static std::vector<double> filter(const std::vector<double>& window, const std::vector<double>& signal)
{
	auto filtered_signal = signal;
	return filtered_signal;
}



Neurogram::Neurogram(const size_t n_cf) :
	cfs_(log_space(std::log10(250.0), std::log10(16e3), n_cf)),
	// assumes no hearing loss
	db_loss_(n_cf, 0.0),
	coh_cs_(n_cf, 1.0),
	ihc_cs_(n_cf, 1.0),
	ohc_loss_(n_cf, 0.0),
	an_population_(Neurogram::generate_an_population(n_cf, 10, 10, 30))
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
		generate_fiber_set(n_cf, n_med,MEDIUM, 4.0, 4.0, .2, 18.0),
		generate_fiber_set(n_cf, n_high,HIGH, 70.0, 30, 18.0, 180.)
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

	const auto smw_ft = hamming(32);
	const auto smw_mr = hamming(128);



	for (size_t cf_i = 0; cf_i < cfs_.size(); cf_i++)
	{
		const auto fibers = get_fibers(cf_i);

		const auto ihc = inner_hair_cell(
			sound_wave, cfs_[cf_i], n_rep, time_resolution, rep_time, coh_cs_[cf_i], ihc_cs_[cf_i], species
		);

		const int n_timesteps = static_cast<int>(ihc.size() / n_rep);

		for (const auto& fiber : fibers)
		{
			const auto pla = synapse_mapping::map(ihc,
				fiber.spont,
				cfs_[cf_i],
				sampling_freq,
				time_resolution,
				SOFTPLUS
			);
			auto out = synapse(pla,
				cfs_[cf_i], n_rep,
				n_timesteps,
				time_resolution,
				noise_type,
				power_law,
				fiber.spont,
				fiber.tabs,
				fiber.trel
			);
			auto ptsh_mr = out.psth;
			auto neurogram_ft = filter(smw_ft, out.psth);
			auto neurogram_mr = filter(smw_mr, ptsh_mr);
			break;
		}

		break;
	}
}
