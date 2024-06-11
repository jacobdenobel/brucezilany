#include "resample.h"
#include "synapse_mapping.h"

using SynapseMappingFunction = std::function<double(double)>;

namespace synapse_mapping
{
	double power_map(const double x, const double cf_factor, const double mul_factor)
	{
		double res = pow(10, 0.9 * log10(fabs(x) * cf_factor) + mul_factor);
		if (x < 0) res = -res;
		return res;
	}	

	double none(const double x)
	{
		return x;
	}

	double softplus(const double x)
	{
		constexpr double p2 = 1165;
		constexpr double p1 = 0.00172;

		double mapping_out_k = p1 * (log(1.0 + exp(p2 * x)) - log(2.0));
		if (isfinite(mapping_out_k) < 1)
			mapping_out_k = p1 * (p2 * x - log(2.0)); /* linear asymptote */
		return mapping_out_k;
	}	

	double exponential(const double x) {
		constexpr double p1 = 0.001268;
		constexpr double p2 = 747.9;
		return std::min(p1 * (exp(p2 * x) - 1.0), 30.0);
	}

	double boltzman(const double x)
	{
		constexpr double p1 = 787.77;
		constexpr double p2 = 749.69;
		return 1.0 / (1.0 + p1 * exp(-p2 * x)) - 1.0 / (1.0 + p1);
	}

	SynapseMappingFunction get_function(const SynapseMapping mapping_function)
	{
		switch (mapping_function)
		{
		case SOFTPLUS:
			return softplus;
		case EXPONENTIAL:
			return exponential;
		case BOLTZMAN:
			return boltzman;
		case NONE:
		default:
			return none;
		}
	}

	std::vector<double> map(
		const std::vector<double>& ihc_output,
		const double spontaneous_firing_rate,
		const double cf,
		const double sampling_frequency,
		const double time_resolution,
		const SynapseMapping mapping_function
	)
	{
		const double cf_slope = pow(spontaneous_firing_rate, 0.19) * pow(10, -0.87);
		const double cf_const = 0.1 * pow(log10(spontaneous_firing_rate), 2) + 0.56 * log10(spontaneous_firing_rate) - 0.84;
		const double cf_sat = pow(10, (cf_slope * 8965.5 / 1e3 + cf_const));
		const double cf_factor = std::min(cf_sat, pow(10, cf_slope * cf / 1e3 + cf_const)) * 2.0;
		const double mul_factor = std::max(2.95 * std::max(1.0, 1.5 - spontaneous_firing_rate / 100), 4.3 - 0.2 * cf / 1e3);
		const int n_total_timesteps = static_cast<int>(ihc_output.size());
		const int delay_point = static_cast<int>(floor(7500 / (cf / 1e3)));

		std::vector<double> output_signal(static_cast<long>(ceil(n_total_timesteps + 3 * delay_point)));
		const auto mapper = get_function(mapping_function);

		for (int k = 0; k < static_cast<int>(ceil(n_total_timesteps)); k++)
			output_signal[k + delay_point] = power_map(mapper(ihc_output[k]), cf_factor, mul_factor) + 3.0 * spontaneous_firing_rate;

		for (int k = 0; k < delay_point; k++)
			output_signal[k] = output_signal[delay_point];

		for (int k = n_total_timesteps + delay_point; k < n_total_timesteps + 3 * delay_point; k++)
			output_signal[k] = output_signal[k - 1] + 3.0 * spontaneous_firing_rate;

		const int down_factor = static_cast<int>(ceil(1 / (time_resolution * sampling_frequency)));
		return resample(1, down_factor, output_signal);
	}
}
