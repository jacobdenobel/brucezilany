#include "bruce2018.h"

namespace utils
{
	std::mt19937 GENERATOR;

	template <typename D>
	std::vector<double> random(const size_t n, D& d)
	{
		std::vector<double> r(n);
		for (auto& ri : r)
			ri = d(GENERATOR);
		return r;
	}


	std::vector<double> rand(const size_t n)
	{
		static std::uniform_real_distribution<double> d(0, 1.0);
		return random(n, d);
	}

	double rand1()
	{
		static std::uniform_real_distribution<double> d(0, 1.0);
		return d(GENERATOR);
	}

	std::vector<double> randn(const size_t n)
	{
		static std::normal_distribution<double> d(0, 1.0);
		return random(n, d);
	}

	void fill_gaussian(std::vector<double>& x)
	{
		static std::normal_distribution<double> d(0, 1.0);
		for (auto& xi : x)
			xi = d(GENERATOR);
	}


	static std::vector<double> generate_zmag(const size_t n_samples)
	{
		static std::vector<double> z_mag;

		// Only generate z_mag whenever n_samples changes
		if (z_mag.size() != n_samples)
		{
			const int n_fft = static_cast<int>(std::pow(2, std::ceil(log2(2 * (n_samples - 1)))));
			const size_t n_fft_half = static_cast<size_t>(std::round(n_fft / 2));
			std::valarray<std::complex<double>> fft_data(n_fft);
			z_mag.resize(n_fft);


			std::generate(std::begin(fft_data), std::end(fft_data),
			              [n_fft_half, n = 0, reverse = false]() mutable
			              {
				              if (n + 1 > n_fft_half) reverse = true;
				              const double k = !reverse ? n++ : n--;
				              return 0.5 * (pow(k + 1, 2. * 0.9) - (2.0 * pow(k, 2.0 * 0.9)) +
					              pow(abs(k - 1), 2. * 0.9));
			              });

			fft(fft_data);
			for (size_t i = 0; i < z_mag.size(); ++i)
			{
				if (fft_data[i].real() < 0.0)
				{
					throw(std::runtime_error("FFT produced > 0"));
				}
				z_mag[i] = std::sqrt(fft_data[i].real());
			}
		}
		return z_mag;
	}

	static void fill_noise_vectors(std::vector<double>& zr1, std::vector<double>& zr2, const NoiseType noise)
	{
		switch (noise)
		{
		case ONES:
			zr1.assign(zr1.size(), 1);
			zr2.assign(zr2.size(), 1);
			break;
		case FIXED_MATLAB:
			zr1 = {
				0.539001198446002, -0.333146282212077, 0.758784275258885, -0.960019229100215,
				-2.010902387858044, -0.014145783976321, 0.014846193555120, 0.179719933210648,
				-2.035475594737959, -0.357587732438863, 0.317062418711363, -1.266378348690577,
				1.038708704838524, -2.500059203501081, -1.252332731960022, 1.230339014018892,
				-0.504687908175280, 0.919640621536610, -0.234470350850954, 0.530697743839911,
				0.660825091280324, 0.855468294638247, -0.994629072636940, -2.231455213644026,
				0.318559022665053, 0.632957296094154, -0.151148210794462, -0.816060813871062,
				-1.014897009384865, 0.518977711821625, -0.059474326486106, 0.731639398082223
			};
			zr2 = {
				-0.638409626955796, -0.061701505688751, -0.218192062027145, 0.203235982652021,
				-0.098642410359283, 0.945333174032015, -0.801457072154293, -0.085099820744463,
				0.789397946964058, 1.226327097545239, -0.900142192575332, 0.424849252031244,
				-0.387098269639317, 1.170523150888439, -0.072882198808166, -1.612913245229722,
				-0.702699919458338, -0.283874347267996, 0.450432043543390, -0.259699095922555,
				0.409258053752079, 1.926425247717760, -0.945190729563938, -0.854589093975853,
				-0.219510861979715, 0.449824239893538, 0.257557798875416, 0.212844513926846,
				-0.087690563274934, 0.231624682299529, -0.563183338456413, -1.188876899529859
			};
			break;
		case FIXED_SEED:
			GENERATOR.seed(42);
			[[fallthrough]];
		default:
			fill_gaussian(zr1);
			fill_gaussian(zr2);
		}
	}

	std::vector<double> fast_fractional_gaussian_noise(const int n_out, const NoiseType noise, const int mu)
	{
		// TODO check if n_out can change
		using namespace std::complex_literals;

		constexpr int resample_factor = 1000;

		const size_t n_samples = static_cast<int>(std::max(10.0, std::ceil(n_out / resample_factor) + 1));
		static std::vector<double> y(n_samples);
		static std::vector<double> z_mag = generate_zmag(n_samples);
		static std::valarray<std::complex<double>> z(z_mag.size());
		static std::vector<double> zr1(z_mag.size());
		static std::vector<double> zr2(z_mag.size());

		fill_noise_vectors(zr1, zr2, noise);

		for (size_t i = 0; i < z_mag.size(); i++)
			z[i] = z_mag[i] * (zr1[i] + 1i * zr2[i]);

		ifft(z);

		const double root_n = std::sqrt(z_mag.size());
		for (size_t i = 0; i < n_samples; i++)
			y[i] = z[i].real() * root_n;

		std::vector<double> output_signal(n_out);
		resample(resample_factor, 1, y, output_signal);
		output_signal.resize(n_out);

		const double sigma = mu < .2 ? 1.0 : mu < 20 ? 10 : mu / 2.0;
		for (auto& yi : output_signal)
			yi *= sigma;
		return output_signal;
	}
}
