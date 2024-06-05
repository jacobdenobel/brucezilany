#include "bruce2018.h"

namespace stats
{
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
		syn::SynapseOutput& res,
		const size_t n_rep,
		const size_t n_sites,
		const int totalstim,
		const double absolute_refractory_period, 
		const double relative_refractory_period
	)
	{
		const size_t n = totalstim * n_rep;

		res.mean_relative_refractory_period.resize(n);
		res.mean_firing_rate.resize(n);
		res.variance_firing_rate.resize(n);


		for (int i = 0; i < n; i++)
		{
			const int i_pst = static_cast<int>(fmod(i, totalstim));
			if (res.synaptic_output[i] > 0)
			{
				res.mean_relative_refractory_period[i] = std::min(relative_refractory_period * 100 / res.synaptic_output[i],
					relative_refractory_period);
				/* estimated instantaneous mean rate */
				res.mean_firing_rate[i_pst] += res.synaptic_output[i] / (res.synaptic_output[i] * (absolute_refractory_period + res.
					redocking_time[i] / n_sites + res.mean_relative_refractory_period[i]) + 1) / n_rep;

				res.variance_firing_rate[i_pst] += instantaneous_variance(res.synaptic_output[i], res.redocking_time[i],
					absolute_refractory_period,
					res.mean_relative_refractory_period[i]) / n_rep;
			}
			else
				res.mean_relative_refractory_period[i] = relative_refractory_period;
		}
	}
}
