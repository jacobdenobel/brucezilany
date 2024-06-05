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
#include <iomanip>

#include "bruce2018.h"


syn::SynapseOutput synapse(
	std::vector<double>& sampIHC, // resampled powerlaw mapping of ihc output, see map_to_power_law
	const double cf,
	const size_t nrep,
	const int totalstim,
	const double time_resolution, // tdres
	const NoiseType noise, // NoiseType
	const PowerLaw approximate, // implnt
	const double spont_rate, // spnt
	const double abs_refractory_period, // tabs
	const double rel_refractory_period, // trel,
	const bool calculate_stats
)
{
	utils::validate_parameter(spont_rate, 1e-4, 180., "spont_rate");
	utils::validate_parameter(nrep, size_t{0}, std::numeric_limits<size_t>::max(), "nrep");
	utils::validate_parameter(abs_refractory_period, 0., 20e-3, "abs_refractory_period");
	utils::validate_parameter(rel_refractory_period, 0., 20e-3, "rel_refractory_period");

	auto res = syn::SynapseOutput(nrep, totalstim);

	///*====== Run the synapse model ======*/
	syn::synapse(sampIHC, time_resolution, cf, spont_rate, noise, approximate, res);


	///*======  Synaptic Release/Spike Generation Parameters ======*/
	const int n_sites = 4; /* Number of synpatic release sites */
	constexpr double t_rd_rest = 14.0e-3; /* Resting value of the mean redocking time */
	constexpr double t_rd_jump = 0.4e-3; /* Size of jump in mean redocking time when a redocking event occurs */
	const double t_rd_init = t_rd_rest + 0.02e-3 * spont_rate - t_rd_jump; /* Initial value of the mean redocking time */
	constexpr double tau = 60.0e-3; /* Time constant for short-term adaptation (in mean redocking time) */


	const int n_spikes = syn::SpikeGenerator(res.synaptic_output.data(), time_resolution, t_rd_rest, t_rd_init, tau, t_rd_jump, n_sites,
		abs_refractory_period, rel_refractory_period, spont_rate, totalstim, nrep, res.spike_times,
		res.redocking_time.data());

	/* Generate PSTH */
	for (int i = 0; i < n_spikes; i++)
		res.psth[static_cast<int>(fmod(res.spike_times[i], time_resolution * totalstim) / time_resolution)]++;

	if (calculate_stats)
		stats::calculate_refractory_and_redocking_stats(res, nrep, n_sites, totalstim, abs_refractory_period, rel_refractory_period);

	return res;
}

namespace syn
{
	void synapse(
		const std::vector<double>& amplitude_ihc,
		const double time_resolution,
		const double cf,
		const double spontaneous_firing_rate,
		const NoiseType noise,
		const PowerLaw power_law,
		SynapseOutput& res
	)
	{
		constexpr double sampling_frequency = 10e3 /* Sampling frequency used in the synapse */;
		const int resampling_size = static_cast<int>(ceil(1 / (time_resolution * sampling_frequency)));
		const int delay_point = static_cast<int>(floor(7500 / (cf / 1e3)));
		const int n = static_cast<int>(floor((static_cast<int>(res.n_total_timesteps) + 2 * delay_point) * time_resolution * sampling_frequency));

		const auto pla_out = pla::power_law(amplitude_ihc, noise, spontaneous_firing_rate, sampling_frequency, power_law, n);
			
		/*---------------------------------------------------------*/
		/*----Up sampling to original (High 100 kHz) sampling rate-*/
		/*---------------------------------------------------------*/
		for (int z = delay_point / resampling_size; z < n - 1; ++z)
		{
			const double incr = (pla_out[z + 1] - pla_out[z]) / resampling_size;
			for (int b = 0; b < resampling_size; ++b)
			{
				const size_t resampled_index = std::max(z * resampling_size + b - delay_point, 0);
				if (resampled_index >= res.n_total_timesteps) break;
				res.synaptic_output[resampled_index] = pla_out[z] + b * incr;
			}
		}	
	}

	/* ------------------------------------------------------------------------------------ */
	/* Pass the output of synapse model through the Spike Generator */
	/* ------------------------------------------------------------------------------------ */
	int SpikeGenerator(double* synout, double tdres, double t_rd_rest, double t_rd_init, double tau, double t_rd_jump,
	                   int nSites, double tabs, double trel, double spont, int totalstim, int nrep,
	                   std::vector<double>& spike_times, double* trd_vector)
	{
		auto preRelease_initialGuessTimeBins = std::vector<double>(nSites);
		auto elapsed_time = std::vector<double>(nSites);
		auto previous_release_times = std::vector<double>(nSites);
		auto current_release_times = std::vector<double>(nSites);
		auto oneSiteRedock = std::vector<double>(nSites);
		auto Xsum = std::vector<double>(nSites);
		auto unitRateInterval = std::vector<double>(nSites);

		/* Initial < redocking time associated to nSites release sites */
		for (int i = 0; i < nSites; i++)
			oneSiteRedock[i] = -t_rd_init * log(utils::rand1());

		/* Initial  preRelease_initialGuessTimeBins  associated to nsites release sites */
		std::vector<double> preReleaseTimeBinsSorted(nSites);
		for (int i = 0; i < nSites; i++)
		{
			preRelease_initialGuessTimeBins[i] = std::max(static_cast<double>(-totalstim * nrep),
			                                              ceil((nSites / std::max(synout[0], 0.1) + t_rd_init) * log(
				                                              utils::rand1()) / tdres));
			preReleaseTimeBinsSorted[i] = preRelease_initialGuessTimeBins[i];
		}

		// Now Sort the four initial preRelease times and associate
		// the farthest to zero as the site which has also generated a spike 
		std::sort(preReleaseTimeBinsSorted.begin(), preReleaseTimeBinsSorted.end());

		/* Consider the inital previous_release_times to be  the preReleaseTimeBinsSorted *tdres */
		for (int i = 0; i < nSites; i++)
			previous_release_times[i] = preReleaseTimeBinsSorted[i] * tdres;


		/* The position of first spike, also where the process is started- continued from the past */
		int kInit = static_cast<int>(preReleaseTimeBinsSorted[0]);

		/* Current refractory time */
		double Tref = tabs - trel * log(utils::rand1());

		/*initlal refractory regions */
		double current_refractory_period = static_cast<double>(kInit) * tdres;

		int spCount = 0; /* total numebr of spikes fired */
		int k = kInit; /*the loop starts from kInit */

		/* set dynamic mean redocking time to initial mean redocking time  */
		double previous_redocking_period = t_rd_init;
		double current_redocking_period = previous_redocking_period;
		int t_rd_decay = 1;
		/* Logical "true" as to whether to decay the value of current_redocking_period at the end of the time step */
		int rd_first = 0; /* Logical "false" as to whether to a first redocking event has occurred */

		/* a loop to find the spike times for all the totalstim*nrep */
		while (k < totalstim * nrep)
		{
			for (int siteNo = 0; siteNo < nSites; siteNo++)
			{
				if (k > preReleaseTimeBinsSorted[siteNo])
				{
					/* redocking times do not necessarily occur exactly at time step value - calculate the
					 * number of integer steps for the elapsed time and redocking time */
					const int oneSiteRedock_rounded = static_cast<int>(oneSiteRedock[siteNo] / tdres);
					const int elapsed_time_rounded = static_cast<int>(elapsed_time[siteNo] / tdres);
					if (oneSiteRedock_rounded == elapsed_time_rounded)
					{
						/* Jump  trd by t_rd_jump if a redocking event has occurred   */
						current_redocking_period = previous_redocking_period + t_rd_jump;
						previous_redocking_period = current_redocking_period;
						t_rd_decay = 0; /* Don't decay the value of current_redocking_period if a jump has occurred */
						rd_first = 1; /* Flag for when a jump has first occurred */
					}

					/* to be sure that for each site , the code start from its
					 * associated  previus release time :*/
					elapsed_time[siteNo] = elapsed_time[siteNo] + tdres;
				}


				/*the elapsed time passes  the one time redock (the redocking is finished),
				 * In this case the synaptic vesicle starts sensing the input
				 * for each site integration starts after the redockinging is finished for the corresponding site)*/
				if (elapsed_time[siteNo] >= oneSiteRedock[siteNo])
				{
					Xsum[siteNo] = Xsum[siteNo] + synout[std::max(0, k)] / nSites;
					/* There are  nSites integrals each vesicle senses 1/nosites of  the whole rate */
				}

				if ((Xsum[siteNo] >= unitRateInterval[siteNo]) && (k >= preReleaseTimeBinsSorted[siteNo]))
				{
					/* An event- a release  happened for the siteNo*/

					oneSiteRedock[siteNo] = -current_redocking_period * log(utils::rand1());
					current_release_times[siteNo] = previous_release_times[siteNo] + elapsed_time[siteNo];
					elapsed_time[siteNo] = 0;

					if ((current_release_times[siteNo] >= current_refractory_period))
					{
						/* A spike occured for the current event- release
						               * spike_times[(int)(current_release_times[siteNo]/tdres)-kInit+1 ] = 1;*/

						/*Register only non negative spike times */
						if (current_release_times[siteNo] >= 0)
						{
							spike_times.push_back(current_release_times[siteNo]);
							//sptime[spCount] = ; 
							spCount++;
						}

						double trel_k = std::min(trel * 100 / synout[std::max(0, k)], trel);

						Tref = tabs - trel_k * log(utils::rand1()); /*Refractory periods */

						current_refractory_period = current_release_times[siteNo] + Tref;
					}

					previous_release_times[siteNo] = current_release_times[siteNo];

					Xsum[siteNo] = 0;
					unitRateInterval[siteNo] = static_cast<int>(-log(utils::rand1()) / tdres);
				}
			}

			/* Decay the adapative mean redocking time towards the resting value if no redocking events occurred in this time step */
			if ((t_rd_decay == 1) && (rd_first == 1))
			{
				current_redocking_period = previous_redocking_period - (tdres / tau) * (previous_redocking_period -
					t_rd_rest);
				previous_redocking_period = current_redocking_period;
			}
			else
			{
				t_rd_decay = 1;
			}

			/* Store the value of the adaptive mean redocking time if it is within the simulation output period */
			if ((k >= 0) && (k < totalstim * nrep))
				trd_vector[k] = current_redocking_period;

			k++;
		}
		return spCount;
	}
}
