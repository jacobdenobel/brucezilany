#pragma once

#include <functional>
#include <vector>
#include "types.h"

using SynapseMappingFunction = std::function<double(double)>;

namespace synapse_mapping
{
	/**
	 * Apply power transformation: 10^(0.9 * log10(|x| * cf_factor) + mul_factor)
	 * @param x the value to transform
	 * @param cf_factor constant
	 * @param mul_factor constant
	 * @return transformed value
	 */
	double power_map(double x, double cf_factor, double mul_factor);

	/**
	 * Identity function
	 * @param x input value
	 * @return x
	 */
	double none(double x);

	/**
	 * Apply softplus function, with constants 0.00172 and 1165
	 * @param x input value
	 * @return softplus(x)
	 */
	double softplus(double x);

	/**
	 * Apply scaled exponential function to x
	 * @param x input value
	 * @return exp(x)
	 */
	double exponential(double x);

	/**
	 * Apply scaled boltzman function to x
	 * @param x input value
	 * @return boltzman(x)
	 */
	double boltzman(double x);

	/**
	 * Select appropriate synapse mapping function
	 * @param mapping_function the type of mapping function
	 * @return a function double -> double
	 */
	SynapseMappingFunction get_function(SynapseMapping mapping_function);

	/**
	 *	Synapse mapping function. Maps the output of the inner hair cell model (ihc_output)
	 *	to a rescaled output, using a given non-linear mapping function.
	 *
	 * @param ihc_output the output of the inner hair cell model
	 * @param spontaneous_firing_rate the spontaneous firing rate
	 * @param characteristic_frequency the characteristic frequency
	 * @param time_resolution the time resolution of the input
	 * @param mapping_function Type of mapping function to be used
	 * @return transformed inner hair cell output
	 */
	std::vector<double> map(
		const std::vector<double>& ihc_output,
		double spontaneous_firing_rate,
		double characteristic_frequency,
		double time_resolution,
		SynapseMapping mapping_function
	);
}

