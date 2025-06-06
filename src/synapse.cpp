/* This is the BEZ2018a version of the code for auditory periphery model from the Carney, Bruce and Zilany labs.
*
* This release implements the version of the model described in:
*
* Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
* model of the synapse between the inner hair cell and auditory nerve:
* Implications of limited neurotransmitter release sites," Hearing Research 360:40-54.
* (Special Issue on "Computational Models in Hearing".)
*
* with the synapse modifications described in:
*
* Bruce, I., Buller, A., and Zilany, M. "Modeling of auditory nerve fiber input/output functions
* near threshold," Acoustics 2023, Sydney, Australia, December 2023.
*
* Please cite these two publications if you publish any research
* results obtained with this code or any modified versions of this code.
*
* See the file readme.txt for details of compiling and running the model.
*
* %%% Ian C. Bruce (ibruce@ieee.org), Muhammad S. A. Zilany (msazilany@gmail.com)
* - December 2023 %%%
*
*/

#include "bruce.h"


namespace syn
{
	void up_sample_synaptic_output(const std::vector<double>& pla_out, const double time_resolution,
		const double sampling_frequency, const int delay_point, SynapseOutput& res)
	{
		const int resampling_size = static_cast<int>(ceil(1 / (time_resolution * sampling_frequency)));
		/*---------------------------------------------------------*/
		/*----Up sampling to original (High 100 kHz) sampling rate-*/
		/*---------------------------------------------------------*/
		for (int z = delay_point / resampling_size; z < static_cast<int>(pla_out.size()) - 1; ++z)
		{
			const double incr = (pla_out[z + 1] - pla_out[z]) / resampling_size;
			for (int b = 0; b < resampling_size; ++b)
			{
				const int resampled_index = std::max(z * resampling_size + b - delay_point, 0);
				if (resampled_index >= res.n_total_timesteps) break;
				res.synaptic_output[resampled_index] = pla_out[z] + b * incr;
			}
		}
	}


	template <size_t nSites>
	int spike_generator(
		const double time_resolution,
		const double spontaneous_firing_rate,
		const double abs_refractory_period,
		const double rel_refractory_period,
		utils::RandomGenerator& rng,
		SynapseOutput& res
	)
	{
		constexpr double t_rd_rest = 14.0e-3; /* Resting value of the mean redocking time */
		constexpr double t_rd_jump = 0.4e-3; /* Size of jump in mean redocking time when a redocking event occurs */
		constexpr double tau = 60.0e-3; /* Time constant for short-term adaptation (in mean redocking time) */
		const double t_rd_init = t_rd_rest + 0.02e-3 * spontaneous_firing_rate - t_rd_jump;
		/* Initial value of the mean redocking time */

		std::array<double, nSites> elapsed_time{};
		std::array<double, nSites> previous_release_times{};
		std::array<double, nSites> previous_release_times_bins{};
		std::array<double, nSites> current_release_times{};
		std::array<double, nSites> one_site_redocking{};
		std::array<double, nSites> x_sum{};
		std::array<double, nSites> unit_rate_interval{};


		/* Initial  preRelease_initialGuessTimeBins associated to nsites release sites */
		for (size_t i = 0; i < nSites; i++)
		{
			one_site_redocking[i] = -t_rd_init * log(rng.rand1());
			previous_release_times_bins[i] = std::max(static_cast<double>(-res.n_total_timesteps),
				ceil((nSites / std::max(res.synaptic_output[0], 0.1) + t_rd_init)
					* log(rng.rand1()) / time_resolution));
		}

		std::sort(previous_release_times_bins.begin(), previous_release_times_bins.end());

		/* Consider the initial previous_release_times to be  the preReleaseTimeBinsSorted *time_resolution */
		for (size_t i = 0; i < nSites; i++)
			previous_release_times[i] = previous_release_times_bins[i] * time_resolution;


		/* The position of first spike, also where the process is started - continued from the past */
		const int k_init = static_cast<int>(previous_release_times_bins[0]);

		/* Current refractory time */
		double t_ref = abs_refractory_period - rel_refractory_period * log(rng.rand1());

		/*initial refractory regions */
		double current_refractory_period = static_cast<double>(k_init) * time_resolution;

		int spike_count = 0;

		/* set dynamic mean redocking time to initial mean redocking time  */
		double previous_redocking_period = t_rd_init;
		double current_redocking_period = previous_redocking_period;
		int t_rd_decay = 1;

		/* Logical "true" whether to decay the value of current_redocking_period at the end of the time step */
		int rd_first = 0; /* Logical "false" whether to a first redocking event has occurred */

		for (int k = k_init; k < res.n_total_timesteps; ++k)
		{
			for (size_t site_no = 0; site_no < nSites; site_no++)
			{
				if (k > previous_release_times_bins[site_no])
				{
					/* redocking times do not necessarily occur exactly at time step value - calculate the
					 * number of integer steps for the elapsed time and redocking time */
					const int one_site_redocking_rounded = static_cast<int>(one_site_redocking[site_no] /
						time_resolution);
					const int elapsed_time_rounded = static_cast<int>(elapsed_time[site_no] / time_resolution);
					if (one_site_redocking_rounded == elapsed_time_rounded)
					{
						/* Jump  trd by t_rd_jump if a redocking event has occurred   */
						current_redocking_period = previous_redocking_period + t_rd_jump;
						previous_redocking_period = current_redocking_period;
						t_rd_decay = 0; /* Don't decay the value of current_redocking_period if a jump has occurred */
						rd_first = 1; /* Flag for when a jump has first occurred */
					}

					/* to be sure that for each site , the code start from its
					 * associated previous release time :*/
					elapsed_time[site_no] = elapsed_time[site_no] + time_resolution;
				}


				/*the elapsed time passes  the one time redocking (the redocking is finished),
				 * In this case the synaptic vesicle starts sensing the input
				 * for each site integration starts after the redocking is finished for the corresponding site)*/
				if (elapsed_time[site_no] >= one_site_redocking[site_no])
				{
					x_sum[site_no] = x_sum[site_no] + res.synaptic_output[std::max(0, k)] / nSites;
					/* There are  nSites integrals each vesicle senses 1/ nosites of  the whole rate */
				}

				if ((x_sum[site_no] >= unit_rate_interval[site_no]) && (k >= previous_release_times_bins[site_no]))
				{
					/* An event- a release  happened for the siteNo*/

					one_site_redocking[site_no] = -current_redocking_period * log(rng.rand1());
					current_release_times[site_no] = previous_release_times[site_no] + elapsed_time[site_no];
					elapsed_time[site_no] = 0;

					if ((current_release_times[site_no] >= current_refractory_period))
					{
						if (current_release_times[site_no] >= 0)
						{
							const auto spike_time = current_release_times[site_no];
							const auto spike_index = static_cast<int>(fmod(
								spike_time, time_resolution * res.n_timesteps) / time_resolution);
							res.spike_times.push_back(spike_time);
							++res.psth[spike_index];
							spike_count++;
						}

						const double t_rel_k = std::min(
							rel_refractory_period * 100 / res.synaptic_output[std::max(0, k)], rel_refractory_period);

						t_ref = abs_refractory_period - t_rel_k * log(rng.rand1());

						current_refractory_period = current_release_times[site_no] + t_ref;
					}

					previous_release_times[site_no] = current_release_times[site_no];

					x_sum[site_no] = 0;
					unit_rate_interval[site_no] = static_cast<int>(-log(rng.rand1()) / time_resolution);
				}
			}

			/* Decay the adaptive mean redocking time towards the resting value if no redocking events occurred in this time step */
			if ((t_rd_decay == 1) && (rd_first == 1))
			{
				current_redocking_period = previous_redocking_period - (time_resolution / tau) * (
					previous_redocking_period -
					t_rd_rest);
				previous_redocking_period = current_redocking_period;
			}
			else
			{
				t_rd_decay = 1;
			}

			/* Store the value of the adaptive mean redocking time if it is within the simulation output period */
			res.redocking_time[std::max(k, 0)] = current_redocking_period;
		}
		return spike_count;
	}

	double instantaneous_variance(const double synaptic_output, const double redocking_time, const double absolute_refractory_period, const double relative_refractory_period)
	{
		const double s2 = synaptic_output * synaptic_output;
		const double s3 = s2 * synaptic_output;
		const double s4 = s3 * synaptic_output;
		const double s5 = s4 * synaptic_output;
		const double s6 = s5 * synaptic_output;
		const double s7 = s6 * synaptic_output;
		const double s8 = s7 * synaptic_output;
		const double trel2 = relative_refractory_period * relative_refractory_period;
		const double t2 = redocking_time * redocking_time;
		const double t3 = t2 * redocking_time;
		const double t4 = t3 * redocking_time;
		const double t5 = t4 * redocking_time;
		const double t6 = t5 * redocking_time;
		const double t7 = t6 * redocking_time;
		const double t8 = t7 * redocking_time;
		const double st = (synaptic_output * redocking_time + 4);
		const double st4 = st * st * st * st;
		const double ttts = redocking_time / 4 + absolute_refractory_period + relative_refractory_period + 1 / synaptic_output;
		const double ttts3 = ttts * ttts * ttts;

		const double numerator = (11 * s7 * t7) / 2 + (3 * s8 * t8) / 16 + 12288 * s2
			* trel2 + redocking_time * (22528 * s3 * trel2 + 22528 * synaptic_output)
			+ t6 * (3 * s8 * trel2 + 82 * s6) + t5 * (88 * s7 * trel2 + 664 * s5) + t4
			* (976 * s6 * trel2 + 3392 * s4) + t3 * (5376 * s5 * trel2 + 10624 * s3)
			+ t2 * (15616 * s4 * trel2 + 20992 * s2) + 12288;
		const double denominator = s2 * st4 * (3 * s2 * t2 + 40 * synaptic_output * redocking_time + 48) * ttts3;
		return numerator / denominator;
	}


	void calculate_refractory_and_redocking_stats(
		const int n_sites,
		const double abs_refractory_period,
		const double rel_refractory_period,
		SynapseOutput& res
	)
	{

		res.mean_relative_refractory_period.resize(res.n_total_timesteps);
		res.mean_firing_rate.resize(res.n_total_timesteps);
		res.variance_firing_rate.resize(res.n_total_timesteps);


		for (int i = 0; i < res.n_total_timesteps; i++)
		{
			const int i_pst = static_cast<int>(fmod(i, res.n_timesteps));
			if (res.synaptic_output[i] > 0)
			{
				res.mean_relative_refractory_period[i] = std::min(rel_refractory_period * 100 / res.synaptic_output[i],
					rel_refractory_period);
				/* estimated instantaneous mean rate */
				res.mean_firing_rate[i_pst] += res.synaptic_output[i] / (res.synaptic_output[i] * (abs_refractory_period + res.
					redocking_time[i] / n_sites + res.mean_relative_refractory_period[i]) + 1) / res.n_rep;

				res.variance_firing_rate[i_pst] += instantaneous_variance(res.synaptic_output[i], res.redocking_time[i],
					abs_refractory_period,
					res.mean_relative_refractory_period[i]) / res.n_rep;
			}
			else
				res.mean_relative_refractory_period[i] = rel_refractory_period;
		}
	}
}


syn::SynapseOutput synapse(
	const std::vector<double>& amplitude_ihc, // resampled power law mapping of ihc output, see map_to_synapse
	const double cf,
	const int n_rep,
	const size_t n_timesteps,
	const double time_resolution, // tdres
	const NoiseType noise, // NoiseType
	const PowerLaw pla_impl, // implnt
	const double spontaneous_firing_rate, // spnt
	const double abs_refractory_period, // tabs
	const double rel_refractory_period, // trel,
	const bool calculate_stats,
	std::optional<utils::RandomGenerator> rng
)
{
	utils::validate_parameter(spontaneous_firing_rate, 1e-4, 180., "spontaneous_firing_rate");
	utils::validate_parameter(n_rep, 0, std::numeric_limits<int>::max(), "n_rep");
	utils::validate_parameter(abs_refractory_period, 0., 20e-3, "abs_refractory_period");
	utils::validate_parameter(rel_refractory_period, 0., 20e-3, "rel_refractory_period");

	auto rng_gen = rng.value_or(utils::RandomGenerator(utils::SEED));

	auto res = syn::SynapseOutput(n_rep, static_cast<int>(n_timesteps));

	///*====== Run the synapse model ======*/
	constexpr double sampling_frequency = 10e3 /* Sampling frequency used in the synapse */;
	const int delay_point = static_cast<int>(floor(7500 / (cf / 1e3)));


	const auto pla_out = pla::power_law(
		amplitude_ihc, noise, pla_impl, 
		spontaneous_firing_rate, sampling_frequency,
		delay_point, time_resolution, res.n_total_timesteps, rng_gen);

	up_sample_synaptic_output(pla_out, time_resolution, sampling_frequency, delay_point, res);


	///*======  Synaptic Release/Spike Generation Parameters ======*/
	constexpr int n_sites = 4; /* Number of synaptic release sites */
	const int n_spikes = syn::spike_generator<n_sites>(time_resolution, spontaneous_firing_rate, abs_refractory_period,
		rel_refractory_period, rng_gen, res);

	if (calculate_stats)
		calculate_refractory_and_redocking_stats(
			n_sites, abs_refractory_period, rel_refractory_period, res);

	return res;
}
