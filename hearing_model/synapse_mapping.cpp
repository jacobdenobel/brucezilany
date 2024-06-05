#include "bruce2018.h"

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