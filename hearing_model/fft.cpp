#define _USE_MATH_DEFINES
#include <cmath>

#include <complex>
#include <valarray>
#include <iostream>

namespace utils
{
/**  Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
 * 
 * @param x the signal
 */
void fft(std::valarray<std::complex<double>>& x)
{
	// DFT
	size_t k = x.size();
	const double theta_t = M_PI / static_cast<double>(x.size());
	auto phi_t = std::complex<double>(cos(theta_t), -sin(theta_t));
	while (k > 1)
	{
		const size_t n = k;
		k >>= 1;
		phi_t = phi_t * phi_t;
		std::complex<double> T = 1.0L;
		for (size_t l = 0; l < k; l++)
		{
			for (size_t a = l; a < x.size(); a += n)
			{
				const size_t b = a + k;
				std::complex<double> t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phi_t;
		}
	}
	// Decimate
	const size_t m = static_cast<size_t>(log2(x.size()));
	for (size_t a = 0; a < x.size(); a++)
	{
		size_t b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			const std::complex<double> t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
}

// inverse fft (in-place)
void ifft(std::valarray<std::complex<double>>& x)
{
	// conjugate the complex numbers
	x = x.apply(std::conj);

	// forward fft
	fft(x);

	// conjugate the complex numbers again
	x = x.apply(std::conj);

	// scale the numbers
	x /= static_cast<double>(x.size());
}
}
