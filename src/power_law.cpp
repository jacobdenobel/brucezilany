#include "bruce.h"

namespace pla
{
	void approximate(
		const std::vector<double>& amplitude_ihc,
		const std::vector<double>& random_numbers,
		const int n,
		const double alpha1,
		const double alpha2,
		std::vector<double>& synapse_out
	)
	{
		std::array<double, 3> s1{ amplitude_ihc[0] + random_numbers[0] };
		std::array<double, 3> s2{ amplitude_ihc[0] };
		std::array<double, 3> m1{ 0.2 * s1[0] };
		std::array<double, 3> m2{ 0.2 * s1[0] };
		std::array<double, 3> m3{ 0.2 * s1[0] };
		std::array<double, 3> m4{ 0.2 * s1[0] };
		std::array<double, 3> m5{ 0.2 * s1[0] };
		std::array<double, 3> n1{ 1.0e-3 * s2[0] };
		std::array<double, 3> n2{ 1.0e-3 * s2[0] };
		std::array<double, 3> n3{ 1.0e-3 * s2[0] };

		synapse_out[0] = s1[0] + s2[0];

		s1[1] = std::max(0.0, amplitude_ihc[1] + random_numbers[1] - alpha1 * m5[0]);
		s2[1] = std::max(0.0, amplitude_ihc[1] - alpha2 * n3[0]);

		n1[1] = 1.992127932802320 * n1[0] + 1.0e-3 * (s2[1] - 0.994466986569624 * s2[0]);
		n2[1] = 1.999195329360981 * n2[0] + n1[1] - 1.997855276593802 * n1[0];
		n3[1] = -0.798261718183851 * n3[0] + n2[1] + 0.798261718184977 * n2[0];

		m1[1] = 0.491115852967412 * m1[0] + 0.2 * (s1[1] - 0.173492003319319 * s1[0]);
		m2[1] = 1.084520302502860 * m2[0] + m1[1] - 0.803462163297112 * m1[0];
		m3[1] = 1.588427084535629 * m3[0] + m2[1] - 1.416084732997016 * m2[0];
		m4[1] = 1.886287488516458 * m4[0] + m3[1] - 1.830362725074550 * m3[0];
		m5[1] = 1.989549282714008 * m5[0] + m4[1] - 1.983165053215032 * m4[0];

		synapse_out[1] = s1[1] + s2[1];

		for (int k = 2; k < n; k++)
		{
			const size_t i0 = k % 3;
			const size_t i1 = (k - 1) % 3;
			const size_t i2 = (k - 2) % 3;

			s1[i0] = std::max(0.0, amplitude_ihc[k] + random_numbers[k] - alpha1 * m5[i1]);
			s2[i0] = std::max(0.0, amplitude_ihc[k] - alpha2 * n3[i1]);

			n1[i0] = 1.992127932802320 * n1[i1] - 0.992140616993846 * n1[i2] + 1.0e-3 * (s2[i0] - 0.994466986569624 * s2[i1] + 0.000000000002347 * s2[i2]);
			n2[i0] = 1.999195329360981 * n2[i1] - 0.999195402928777 * n2[i2] + n1[i0] - 1.997855276593802 * n1[i1] + 0.997855827934345 * n1[i2];
			n3[i0] = -0.798261718183851 * n3[i1] - 0.199131619873480 * n3[i2] + n2[i0] + 0.798261718184977 * n2[i1] + 0.199131619874064 * n2[i2];

			m1[i0] = 0.491115852967412 * m1[i1] - 0.055050209956838 * m1[i2] + 0.2 * (s1[i0] - 0.173492003319319 * s1[i1] + 0.000000172983796 * s1[i2]);
			m2[i0] = 1.084520302502860 * m2[i1] - 0.288760329320566 * m2[i2] + m1[i0] - 0.803462163297112 * m1[i1] + 0.154962026341513 * m1[i2];
			m3[i0] = 1.588427084535629 * m3[i1] - 0.628138993662508 * m3[i2] + m2[i0] - 1.416084732997016 * m2[i1] + 0.496615555008723 * m2[i2];
			m4[i0] = 1.886287488516458 * m4[i1] - 0.888972875389923 * m4[i2] + m3[i0] - 1.830362725074550 * m3[i1] + 0.836399964176882 * m3[i2];
			m5[i0] = 1.989549282714008 * m5[i1] - 0.989558985673023 * m5[i2] + m4[i0] - 1.983165053215032 * m4[i1] + 0.983193027347456 * m4[i2];

			synapse_out[k] = s1[i0] + s2[i0];
		}
	}

	void actual(
		const std::vector<double>& amplitude_ihc,
		const std::vector<double>& random_numbers,
		const int n,
		const double alpha1,
		const double alpha2,
		const double beta1,
		const double beta2,
		const double bin_width,
		std::vector<double>& synapse_out
	)
	{
		double i1 = 0, i2 = 0;
		auto s1 = std::vector<double>(n);
		auto s2 = std::vector<double>(n);
		for (int k = 0; k < n; k++)
		{
			s1[k] = std::max(0.0, amplitude_ihc[k] + random_numbers[k] - alpha1 * i1);
			s2[k] = std::max(0.0, amplitude_ihc[k] - alpha2 * i2);

			i1 = 0;
			i2 = 0;
			for (int j = 0; j < k + 1; ++j)
			{
				i1 += (s1[j]) * bin_width / ((k - j) * bin_width + beta1);
				i2 += (s2[j]) * bin_width / ((k - j) * bin_width + beta2);
			}
			synapse_out[k] = s1[k] + s2[k];
		}
	}

	std::vector<double> power_law(
		const std::vector<double>& amplitude_ihc,
		const NoiseType noise,
		const PowerLaw impl,
		const double spontaneous_firing_rate,
		const double sampling_frequency,
		const double delay_point,
		const double time_resolution,
		const int n_total_timesteps,
		utils::RandomGenerator& rng
	)
	{
		const double bin_width = 1 / sampling_frequency;
		constexpr double alpha1 = 1.5e-6 * 100e3;
		constexpr double alpha2 = 1e-2 * 100e3;
		constexpr double beta1 = 5e-4;
		constexpr double beta2 = 1e-1;

		const int n = static_cast<int>(floor((n_total_timesteps + 2.0 * delay_point) * time_resolution * sampling_frequency));
		const auto random_numbers = utils::fast_fractional_gaussian_noise(rng, n, noise, spontaneous_firing_rate);

		std::vector<double> synapse_out(n);
		if (impl == APPROXIMATED)
			approximate(amplitude_ihc, random_numbers, n, alpha1, alpha2, synapse_out);
		else
			actual(amplitude_ihc, random_numbers, n, alpha1, alpha2, beta1, beta2, bin_width, synapse_out);

		return synapse_out;
	}
}
