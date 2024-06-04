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
#include "bruce2018.h"



//! estimated instananeous variance in the discharge rate
static double instanious_variance(const double synout_i, const double trd_vector_i, const double tabs, const double trel_i)
{
	const double s2 = synout_i * synout_i;
	const double s3 = s2 * synout_i;
	const double s4 = s3 * synout_i;
	const double s5 = s4 * synout_i;
	const double s6 = s5 * synout_i;
	const double s7 = s6 * synout_i;
	const double s8 = s7 * synout_i;
	const double trel2 = trel_i * trel_i;
	const double t2 = trd_vector_i * trd_vector_i;
	const double t3 = t2 * trd_vector_i;
	const double t4 = t3 * trd_vector_i;
	const double t5 = t4 * trd_vector_i;
	const double t6 = t5 * trd_vector_i;
	const double t7 = t6 * trd_vector_i;
	const double t8 = t7 * trd_vector_i;
	const double st = (synout_i * trd_vector_i + 4);
	const double st4 = st * st * st * st;
	const double ttts = trd_vector_i / 4 + tabs + trel_i + 1 / synout_i;
	const double ttts3 = ttts * ttts * ttts;

	const double numerator = (11 * s7 * t7) / 2 + (3 * s8 * t8) / 16 + 12288 * s2
		* trel2 + trd_vector_i * (22528 * s3 * trel2 + 22528 * synout_i)
		+ t6 * (3 * s8 * trel2 + 82 * s6) + t5 * (88 * s7 * trel2 + 664 * s5) + t4
		* (976 * s6 * trel2 + 3392 * s4) + t3 * (5376 * s5 * trel2 + 10624 * s3)
		+ t2 * (15616 * s4 * trel2 + 20992 * s2) + 12288;
	const double denominator = s2 * st4 * (3 * s2 * t2 + 40 * synout_i * trd_vector_i + 48) * ttts3;
	return numerator / denominator;
}


syn::SynapseOutput synapse(
	std::vector<double>& sampIHC, // resampled powerlaw mapping of ihc output, see map_to_power_law
	const double cf,
	const size_t nrep,
	const int totalstim,
	const double time_resolution, // tdres
	const NoiseType noise, // NoiseType
	const bool approximate, // implnt
	const double spont_rate, // spnt
	const double abs_refractory_period, // tabs
	const double rel_refractory_period // trel,
)
{
	utils::validate_parameter(spont_rate, 1e-4, 180., "spont_rate");
	utils::validate_parameter(nrep, size_t{0}, std::numeric_limits<size_t>::max(), "nrep");
	utils::validate_parameter(abs_refractory_period, 0., 20e-3, "abs_refractory_period");
	utils::validate_parameter(rel_refractory_period, 0., 20e-3, "rel_refractory_period");

	auto res = syn::SynapseOutput(totalstim, totalstim * nrep);

	///*====== Run the synapse model ======*/
	const double I = syn::Synapse(sampIHC, time_resolution, cf, totalstim, nrep, spont_rate, noise, !approximate, res.output_rate.data());

	///* Calculate the overall mean synaptic rate */
	double total_mean_rate = 0;
	for (int i = 0; i < I; i++)
		total_mean_rate = total_mean_rate + res.output_rate[i] / I;

	///*======  Synaptic Release/Spike Generation Parameters ======*/
	const int n_sites = 4; /* Number of synpatic release sites */
	constexpr double t_rd_rest = 14.0e-3; /* Resting value of the mean redocking time */
	constexpr double t_rd_jump = 0.4e-3; /* Size of jump in mean redocking time when a redocking event occurs */
	const double t_rd_init = t_rd_rest + 0.02e-3 * spont_rate - t_rd_jump; /* Initial value of the mean redocking time */
	constexpr double tau = 60.0e-3; /* Time constant for short-term adaptation (in mean redocking time) */


	///* We register only the spikes at times after zero, the sufficient array size (more than 99.7% of cases) to register spike times  after zero is :/
	// * /*MaxN=signalLengthInSec/meanISI+ 3*sqrt(signalLengthInSec/MeanISI)= nSpike_average +3*sqrt(nSpike_average)*/
	const double mean_isi = (1 / total_mean_rate) + (t_rd_init) / n_sites + abs_refractory_period + rel_refractory_period;
	const double signal_length = totalstim * nrep * time_resolution;
	long max_array_size_spikes = static_cast<long>(ceil(
		static_cast<long>(signal_length / mean_isi + 3 * sqrt(signal_length / mean_isi))));

	std::vector<double> spike_times;
	const int n_spikes = syn::SpikeGenerator(res.output_rate.data(), time_resolution, t_rd_rest, t_rd_init, tau, t_rd_jump, n_sites,
		abs_refractory_period, rel_refractory_period, spont_rate, totalstim, nrep, total_mean_rate, spike_times,
		res.mean_redocking_time.data());

	///* Calculate the analytical estimates of meanrate and varrate and wrapping them up based on no. of repetitions */
	for (int i = 0; i < I; i++)
	{
		const int i_pst = static_cast<int>(fmod(i, totalstim));
		if (res.output_rate[i] > 0)
		{
			res.mean_relative_refractory_period[i] = std::min(rel_refractory_period * 100 / res.output_rate[i], rel_refractory_period);
			/* estimated instantaneous mean rate */
			res.mean_firing_rate[i_pst] += res.output_rate[i] / (res.output_rate[i] * (abs_refractory_period + res.mean_redocking_time[i] / n_sites + res.mean_relative_refractory_period[i]) + 1) / nrep;

			res.variance_firing_rate[i_pst] += instanious_variance(res.output_rate[i], res.mean_redocking_time[i], abs_refractory_period, res.mean_relative_refractory_period[i]) / nrep;
		}
		else
			res.mean_relative_refractory_period[i] = rel_refractory_period;
	}

	/* Generate PSTH */
	for (int i = 0; i < n_spikes; i++)
		res.psth[static_cast<int>(fmod(spike_times[i], time_resolution * totalstim) / time_resolution)]++;
	return res;
}


namespace synapse_mapping
{
	static void none(double* ihcout, std::vector<double>& mappingOut, double spont, double cf, int totalstim, int nrep,
	                 double cf_factor, double multFac)
	{
		for (int k = 0; k < totalstim * nrep; ++k)
		{
			mappingOut[k] = pow(10, (0.9 * log10(fabs(ihcout[k]) * cf_factor)) + multFac);
			if (ihcout[k] < 0) mappingOut[k] = -mappingOut[k];
		}
	}

	static void softplus(double* ihcout, std::vector<double>& mappingOut, double spont, double cf, int totalstim,
	                     int nrep, double cf_factor, double multFac)
	{
		constexpr double p1sp = 0.00172;
		constexpr double p2sp = 1165;

		for (int k = 0; k < totalstim * nrep; ++k)
		{
			double explikeOut = p1sp * (log(1.0 + exp(p2sp * ihcout[k])) - log(2.0));
			if (isfinite(explikeOut) < 1) /* if expOut is infinite, then replace with finite value */
				explikeOut = p1sp * (p2sp * ihcout[k] - log(2.0)); /* linear asymptote */

			mappingOut[k] = pow(10, (0.9 * log10(fabs(explikeOut) * cf_factor)) + multFac);

			if (ihcout[k] < 0) mappingOut[k] = -mappingOut[k];
		}
	}

	static void exponential(double* ihcout, std::vector<double>& mappingOut, double spont, double cf, int totalstim,
	                        int nrep, double cf_factor, double multFac)
	{
		constexpr double p1exp = 0.001268;
		constexpr double p2exp = 747.9;
		for (int k = 0; k < totalstim * nrep; ++k)
		{
			double explikeOut = p1exp * (exp(p2exp * ihcout[k]) - 1.0);
			if (explikeOut > 30) /* if expOut is too large, then replace with reasonable finite */
				explikeOut = 30; /* reasonable finite value */

			mappingOut[k] = pow(10, (0.9 * log10(fabs(explikeOut) * cf_factor)) + multFac);

			if (ihcout[k] < 0) mappingOut[k] = -mappingOut[k];
		}
	}

	static void boltzman(double* ihcout, std::vector<double>& mappingOut, double spont, double cf, int totalstim,
	                     int nrep, double cf_factor, double multFac)
	{
		constexpr double p1bltz = 787.77;
		constexpr double p2bltz = 749.69;
		for (int k = 0; k < totalstim * nrep; ++k)
		{
			double explikeOut = 1.0 / (1.0 + p1bltz * exp(-p2bltz * ihcout[k])) - 1.0 / (1.0 + p1bltz);

			mappingOut[k] = pow(10, (0.9 * log10(fabs(explikeOut) * cf_factor)) + multFac);

			if (ihcout[k] < 0) mappingOut[k] = -mappingOut[k];
		}
	}
}


std::vector<double> map_to_power_law(double* ihcout, double spont, double cf, int totalstim, int nrep, double sampFreq,
                                     double tdres, const SynapseMappingFunction mapping_function)
{
	/*----------------------------------------------------------*/
	/*----- Mapping Function from IHCOUT to input to the PLA ---*/
	/*----------------------------------------------------------*/
	const int resamp = static_cast<int>(ceil(1 / (tdres * sampFreq)));
	const double cfslope = pow(spont, 0.19) * pow(10, -0.87);
	const double cfconst = 0.1 * pow(log10(spont), 2) + 0.56 * log10(spont) - 0.84;
	const double cfsat = pow(10, (cfslope * 8965.5 / 1e3 + cfconst));
	const double cf_factor = std::min(cfsat, pow(10, cfslope * cf / 1e3 + cfconst)) * 2.0;
	const double multFac = std::max(2.95 * std::max(1.0, 1.5 - spont / 100), 4.3 - 0.2 * cf / 1e3);
	const int delaypoint = static_cast<int>(floor(7500 / (cf / 1e3)));

	std::vector<double> mappingOut(static_cast<long>(ceil(totalstim * nrep)));
	std::vector<double> powerLawIn(static_cast<long>(ceil(totalstim * nrep + 3 * delaypoint)));

	switch (mapping_function)
	{
	case SOFTPLUS:
		synapse_mapping::softplus(ihcout, mappingOut, spont, cf, totalstim, nrep, cf_factor, multFac);
		break;
	case EXPONENTIAL:
		synapse_mapping::exponential(ihcout, mappingOut, spont, cf, totalstim, nrep, cf_factor, multFac);
		break;
	case BOLTZMAN:
		synapse_mapping::boltzman(ihcout, mappingOut, spont, cf, totalstim, nrep, cf_factor, multFac);
		break;
	case NONE:
	default:
		synapse_mapping::none(ihcout, mappingOut, spont, cf, totalstim, nrep, cf_factor, multFac);
		break;
	}


	int k = 0;
	for (k = 0; k < delaypoint; k++)
		powerLawIn[k] = mappingOut[0] + 3.0 * spont;
	for (k = delaypoint; k < totalstim * nrep + delaypoint; k++)
		powerLawIn[k] = mappingOut[k - delaypoint] + 3.0 * spont;
	for (k = totalstim * nrep + delaypoint; k < totalstim * nrep + 3 * delaypoint; k++)
		powerLawIn[k] = powerLawIn[k - 1] + 3.0 * spont;


	/*----------------------------------------------------------*/
	/*------ Downsampling to sampFreq (Low) sampling rate ------*/
	/*----------------------------------------------------------*/
	std::vector<double> sampIHC;
	resample(1, resamp, powerLawIn, sampIHC);

	return sampIHC;
}


namespace syn
{
	void SingleAN(
		const std::vector<double>& sampIHC,
		double cf,
		int nrep,
		double tdres,
		int totalstim,
		NoiseType noiseType,
		int implnt,
		double spont,
		double tabs,
		double trel,
		double* meanrate,
		double* varrate,
		double* psth,
		double* synout,
		double* trd_vector,
		double* trel_vector
	)
	{
		/*====== Run the synapse model ======*/
		const double I = Synapse(sampIHC, tdres, cf, totalstim, nrep, spont, noiseType, implnt, synout);

		/* Calculate the overall mean synaptic rate */
		double total_mean_rate = 0;
		for (int i = 0; i < I; i++)
			total_mean_rate = total_mean_rate + synout[i] / I;

		/*======  Synaptic Release/Spike Generation Parameters ======*/
		int nSites = 4; /* Number of synpatic release sites */
		constexpr double t_rd_rest = 14.0e-3; /* Resting value of the mean redocking time */
		constexpr double t_rd_jump = 0.4e-3; /* Size of jump in mean redocking time when a redocking event occurs */
		const double t_rd_init = t_rd_rest + 0.02e-3 * spont - t_rd_jump; /* Initial value of the mean redocking time */
		constexpr double tau = 60.0e-3; /* Time constant for short-term adaptation (in mean redocking time) */


		/* We register only the spikes at times after zero, the sufficient array size (more than 99.7% of cases) to register spike times  after zero is :/
		 * /*MaxN=signalLengthInSec/meanISI+ 3*sqrt(signalLengthInSec/MeanISI)= nSpike_average +3*sqrt(nSpike_average)*/
		const double MeanISI = (1 / total_mean_rate) + (t_rd_init) / nSites + tabs + trel;
		const double SignalLength = totalstim * nrep * tdres;
		long MaxArraySizeSpikes = static_cast<long>(ceil(
			static_cast<long>(SignalLength / MeanISI + 3 * sqrt(SignalLength / MeanISI))));

		std::vector<double> spike_times;
		const int nspikes = SpikeGenerator(synout, tdres, t_rd_rest, t_rd_init, tau, t_rd_jump, nSites,
		                                   tabs, trel, spont, totalstim, nrep, total_mean_rate, spike_times,
		                                   trd_vector);

		/* Calculate the analytical estimates of meanrate and varrate and wrapping them up based on no. of repetitions */
		for (int i = 0; i < I; i++)
		{
			const int ipst = static_cast<int>(fmod(i, totalstim));
			if (synout[i] > 0)
			{
				trel_vector[i] = std::min(trel * 100 / synout[i], trel);
				/* estimated instantaneous mean rate */
				meanrate[ipst] += synout[i] / (synout[i] * (tabs + trd_vector[i] / nSites + trel_vector[i]) + 1) / nrep;

				varrate[ipst] += instanious_variance(synout[i], trd_vector[i], tabs, trel_vector[i]) / nrep;
			}
			else
				trel_vector[i] = trel;
		}

		/* Generate PSTH */
		for (int i = 0; i < nspikes; i++)
			psth[static_cast<int>(fmod(spike_times[i], tdres * totalstim) / tdres)]++;

	} 

	double max_(const std::vector<double>& x)
	{
		double m = -std::numeric_limits<double>::infinity();
		for (auto& xi : x)
			m = max(xi, m);
		return m;
	}

	int approximate(const std::vector<double>& sampIHC, const std::vector<double>& randNums,
		int n, double alpha1, double alpha2, double beta1, double beta2, double binwidth,
		std::vector<double>& synSampOut) {

		auto m1 = std::vector<double>(n);
		auto m2 = std::vector<double>(n);
		auto m3 = std::vector<double>(n);
		auto m4 = std::vector<double>(n);
		auto m5 = std::vector<double>(n);

		auto n1 = std::vector<double>(n);
		auto n2 = std::vector<double>(n);
		auto n3 = std::vector<double>(n);

		auto sout1 = std::vector<double>(n);
		auto sout2 = std::vector<double>(n);

		double I1 = 0, I2 = 0;
		int k = 0;

		for (int indx = 0; indx < n; indx++)
		{
			sout1[k] = std::max(0.0, sampIHC[indx] + randNums[indx] - alpha1 * I1);
			sout2[k] = std::max(0.0, sampIHC[indx] - alpha2 * I2);

			if (k > 1)
			{
				n1[k] = 1.992127932802320 * n1[k - 1] - 0.992140616993846 * n1[k - 2] + 1.0e-3 * (sout2[k] - 0.994466986569624 * sout2[k - 1] + 0.000000000002347 * sout2[k - 2]);
				n2[k] = 1.999195329360981 * n2[k - 1] - 0.999195402928777 * n2[k - 2] + n1[k] - 1.997855276593802 * n1[k - 1] + 0.997855827934345 * n1[k - 2];
				n3[k] = -0.798261718183851 * n3[k - 1] - 0.199131619873480 * n3[k - 2] + n2[k] + 0.798261718184977 * n2[k - 1] + 0.199131619874064 * n2[k - 2];

				m1[k] = 0.491115852967412 * m1[k - 1] - 0.055050209956838 * m1[k - 2] + 0.2 * (sout1[k] - 0.173492003319319 * sout1[k - 1] + 0.000000172983796 * sout1[k - 2]);
				m2[k] = 1.084520302502860 * m2[k - 1] - 0.288760329320566 * m2[k - 2] + m1[k] - 0.803462163297112 * m1[k - 1] + 0.154962026341513 * m1[k - 2];
				m3[k] = 1.588427084535629 * m3[k - 1] - 0.628138993662508 * m3[k - 2] + m2[k] - 1.416084732997016 * m2[k - 1] + 0.496615555008723 * m2[k - 2];
				m4[k] = 1.886287488516458 * m4[k - 1] - 0.888972875389923 * m4[k - 2] + m3[k] - 1.830362725074550 * m3[k - 1] + 0.836399964176882 * m3[k - 2];
				m5[k] = 1.989549282714008 * m5[k - 1] - 0.989558985673023 * m5[k - 2] + m4[k] - 1.983165053215032 * m4[k - 1] + 0.983193027347456 * m4[k - 2];
			}
			else if (k == 1)
			{
				n1[k] = 1.992127932802320 * n1[k - 1] + 1.0e-3 * (sout2[k] - 0.994466986569624 * sout2[k - 1]);
				n2[k] = 1.999195329360981 * n2[k - 1] + n1[k] - 1.997855276593802 * n1[k - 1];
				n3[k] = -0.798261718183851 * n3[k - 1] + n2[k] + 0.798261718184977 * n2[k - 1];

				m1[k] = 0.491115852967412 * m1[k - 1] + 0.2 * (sout1[k] - 0.173492003319319 * sout1[k - 1]);
				m2[k] = 1.084520302502860 * m2[k - 1] + m1[k] - 0.803462163297112 * m1[k - 1];
				m3[k] = 1.588427084535629 * m3[k - 1] + m2[k] - 1.416084732997016 * m2[k - 1];
				m4[k] = 1.886287488516458 * m4[k - 1] + m3[k] - 1.830362725074550 * m3[k - 1];
				m5[k] = 1.989549282714008 * m5[k - 1] + m4[k] - 1.983165053215032 * m4[k - 1];
			}
			else {
				n1[k] = 1.0e-3 * sout2[k];
				n2[k] = n1[k];
				n3[0] = n2[k];

				m1[k] = 0.2 * sout1[k];
				m2[k] = m1[k];	m3[k] = m2[k];
				m4[k] = m3[k];	m5[k] = m4[k];
			}
			I2 = n3[k];
			I1 = m5[k];

			synSampOut[k] = sout1[k] + sout2[k];
			k++;
		}
		return k;
	}

	int actual(
		const std::vector<double>& sampIHC, const std::vector<double>& randNums,
		int n, double alpha1, double alpha2, double beta1, double beta2, double binwidth,
		std::vector<double>& synSampOut
	) {
		int k = 0;
		double I1 = 0, I2 = 0;
		auto sout1 = std::vector<double>(n);
		auto sout2 = std::vector<double>(n);
		for (int indx = 0; indx < n; indx++)
		{
			sout1[k] = std::max(0.0, sampIHC[indx] + randNums[indx] - alpha1 * I1);
			sout2[k] = std::max(0.0, sampIHC[indx] - alpha2 * I2);

			I1 = 0; I2 = 0;
			for (int j = 0; j < k + 1; ++j)
			{
				I1 += (sout1[j]) * binwidth / ((k - j) * binwidth + beta1);
				I2 += (sout2[j]) * binwidth / ((k - j) * binwidth + beta2);
			}
			synSampOut[k] = sout1[k] + sout2[k];
			k++;
		}
		return k;
	}

	int power_law(const std::vector<double>& sampIHC, const NoiseType noise, const int spont, const double sampling_frequency, const bool implnt, std::vector<double>& synSampOut)
	{
		const int n = static_cast<size_t>(synSampOut.size());
	
		const double bin_width = 1 / sampling_frequency;
		const double alpha1 = 1.5e-6 * 100e3;
		const double alpha2 = 1e-2 * 100e3;
		const double beta1 = 5e-4;
		const double beta2 = 1e-1;
	
		const auto random_numbers = utils::fast_fractional_gaussian_noise(n, noise, spont);

		const int k = implnt == 0
			? approximate(sampIHC, random_numbers, n, alpha1, alpha2, beta1, beta2, bin_width, synSampOut)
			: actual(sampIHC, random_numbers, n, alpha1, alpha2, beta1, beta2, bin_width, synSampOut);
		return k;
	}

	double Synapse(
		const std::vector<double>& sampIHC,
		double tdres,
		double cf,
		int totalstim,
		int nrep,
		double spont,
		NoiseType noiseType,
		int implnt,
		double* synout
	)
	{
		constexpr double sampling_frequency = 10e3 /* Sampling frequency used in the synapse */;
		const int resampling_size = static_cast<int>(ceil(1 / (tdres * sampling_frequency)));
		const int delay_point = static_cast<int>(floor(7500 / (cf / 1e3)));
		const auto n = static_cast<long>(ceil((totalstim * nrep + 2 * delay_point) * tdres * sampling_frequency));

		/*----------------------------------------------------------*/
		/*----- Running Power-law Adaptation -----------------------*/
		/*----------------------------------------------------------*/
		auto synSampOut = std::vector<double>(n);

		const int k = power_law(sampIHC, noiseType, spont, sampling_frequency, implnt, synSampOut);
			
		/*---------------------------------------------------------*/
		/*-----Up sampling to o riinal (High 100 kHz) sampling rate-*/
		/*----------------------------------------------------------*/
		auto TmpSyn = std::vector<double>(static_cast<long>(ceil(totalstim * nrep + 2 * delay_point)));

		for (int z = 0; z < k - 1; ++z)
		{
			const double incr = (synSampOut[z + 1] - synSampOut[z]) / resampling_size;
			for (int b = 0; b < resampling_size; ++b)
				TmpSyn[z * resampling_size + b] = synSampOut[z] + b * incr;
		}

		// Assign output
		for (int i = 0; i < totalstim * nrep; ++i)
			synout[i] = TmpSyn[i + delay_point];

		return ceil(totalstim * nrep);
	}

	/* ------------------------------------------------------------------------------------ */
	/* Pass the output of Synapse model through the Spike Generator */
	/* ------------------------------------------------------------------------------------ */
	int SpikeGenerator(double* synout, double tdres, double t_rd_rest, double t_rd_init, double tau, double t_rd_jump,
	                   int nSites, double tabs, double trel, double spont, int totalstim, int nrep,
	                   double total_mean_rate,
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
