#pragma once

#include <vector>

#include "types.h"
#include "utils.h"


namespace pla
{
	/**
	 * Approximate implementation of the power law mapping
	 * @param amplitude_ihc the input
	 * @param random_numbers source of randomness
	 * @param n the size of the output
	 * @param alpha1 constant
	 * @param alpha2 constant
	 * @param synapse_out the output container
	 */
	void approximate(
		const std::vector<double>& amplitude_ihc,
		const std::vector<double>& random_numbers,
		int n,
		double alpha1,
		double alpha2,
		std::vector<double>& synapse_out
	);

	/**
	 * Actual implementation of the power law mapping
	 *
	 * @param amplitude_ihc the input
	 * @param random_numbers source of randomness
	 * @param n the size of the output
	 * @param alpha1 constant
	 * @param alpha2 constant
	 * @param beta1 constant
	 * @param beta2 constant
	 * @param bin_width 
	 * @param synapse_out the output container
	 */
	void actual(
		const std::vector<double>& amplitude_ihc,
		const std::vector<double>& random_numbers,
		int n,
		double alpha1,
		double alpha2,
		double beta1,
		double beta2,
		double bin_width,
		std::vector<double>& synapse_out
	);

	/**
	 * Implementation of the power law mapping function. Applies either the approximate or actual implementation
	 * given the value of impl.
	 *
	 * @param amplitude_ihc the input
	 * @param noise the type of noise to use 
	 * @param impl the type of power law implementation to use
	 * @param spontaneous_firing_rate the spontaneous firing rate
	 * @param sampling_frequency the sampling frequency of the power law function
	 * @param delay_point the delay point of the model
	 * @param time_resolution the time resolution of the model
	 * @param n_total_timesteps the total number of timesteps of the model
	 * @return Transformed input
	 */
	std::vector<double> power_law(
		const std::vector<double>& amplitude_ihc,
		NoiseType noise,
		PowerLaw impl,
		double spontaneous_firing_rate,
		double sampling_frequency,
		double delay_point,
		double time_resolution,
		int n_total_timesteps,
		utils::RandomGenerator& rng
	);
}

