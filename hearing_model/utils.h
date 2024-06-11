#pragma once
#include <complex>
#include <random>
#include <valarray>
#include <vector>
#include <iostream>

#include "types.h"

namespace utils
{
	/**
	 * Helper to validate the value of a parameter at runtime
	 * @tparam T Type of the parameter
	 * @param p the value of the parameter
	 * @param lb the lower bound
	 * @param ub the upper bound
	 * @param name the name of the parameter
	 */
	template <typename T>
	void validate_parameter(const T p, const T lb, const T ub, const std::string& name = "")
	{
		if (p < lb or p > ub)
		{
			std::ostringstream ss;
			ss << name << "= " << p << " is out of bounds [" << lb << ", " << ub << "]";
			std::cout << ss.str() << std::endl;
			throw std::invalid_argument(ss.str());
		}
	}

	/**
	 * The fast-fourier transform of a signal x
	 * @param x signal to transform
	 */
	void fft(std::valarray<std::complex<double>>& x);

	/**
	 * The inverse fast-fourier transform of a signal x
	 * @param x signal to transform
	 */
	void ifft(std::valarray<std::complex<double>>& x);

	//! The source of randomness
	extern std::mt19937 GENERATOR;

	/**
	 * Generate a single uniform random number in [0, 1)
	 * @return the number
	 */
	double rand1();

	/**
	 * Generate a vector of standard normally distributed random numbers
	 * @param n the size of the vector 
	 * @return the vector with random numbers
	 */
	std::vector<double> randn(size_t n);

	/**
	 * @brief Fast (exact) fractional Gaussian noise and Brownian motion generator for a fixed Hurst
	 * index of .9 and a fixed time resolution (tdres) of 1e-4.
	 *
	 *
	 * @param n_out is the length of the output sequence.
	 * @param noise type of random noise
	 * @param mu the mean of the noise
	 * @return returns a sequence of fractional Gaussian noise with a standard deviation of one.
	 */
	std::vector<double> fast_fractional_gaussian_noise(
		int n_out = 5300,
		NoiseType noise = RANDOM,
		double mu = 100
	);
}

