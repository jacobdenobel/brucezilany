#pragma once

/*
Copyright (c) 2009, Motorola, Inc

All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of Motorola nor the names of its contributors may be
used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
using namespace std;
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdexcept>
#include <complex>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cmath>

template <class S1, class S2, class C>
class Resampler
{
public:
	typedef S1 inputType;
	typedef S2 outputType;
	typedef C coefType;

	Resampler(const int up_rate, const int down_rate, C* coefs, const int coef_count) : up_rate_(up_rate), down_rate_(down_rate), t_(0), x_offset_(0)
		/*
		  The coefficients are copied into local storage in a transposed, flipped
		  arrangement.  For example, suppose upRate is 3, and the input number
		  of coefficients coefCount = 10, represented as h[0], ..., h[9].
		  Then the internal buffer will look like this:
				h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
				   0, h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
				   0, h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
		*/
	{
		padded_coef_count_ = coef_count;
		while (padded_coef_count_ % up_rate_)
		{
			padded_coef_count_++;
		}
		coefs_per_phase_ = padded_coef_count_ / up_rate_;

		transposed_coefs_ = new coefType[padded_coef_count_];
		fill(transposed_coefs_, transposed_coefs_ + padded_coef_count_, 0.);

		state_ = new inputType[coefs_per_phase_ - 1];
		state_end_ = state_ + coefs_per_phase_ - 1;
		fill(state_, state_end_, 0.);

		/* This both transposes, and "flips" each phase, while
		   * copying the defined coefficients into local storage.
		   * There is probably a faster way to do this
		   */
		for (int i = 0; i < up_rate_; ++i)
		{
			for (int j = 0; j < coefs_per_phase_; ++j)
			{
				if (j * up_rate_ + i < coef_count)
					transposed_coefs_[(coefs_per_phase_ - 1 - j) + i * coefs_per_phase_] = coefs[j * up_rate_ + i];
			}
		}
	}

	virtual ~Resampler() {
		delete[] transposed_coefs_;
		delete[] state_;
	}

	int apply(S1* in, int in_count, S2* out, const int out_count) {
		if (out_count < needed_out_count(in_count))
			throw invalid_argument("Not enough output samples");

		// x points to the latest processed input sample
		inputType* x = in + x_offset_;
		outputType* y = out;
		inputType* end = in + in_count;
		while (x < end)
		{
			outputType acc = 0.;
			coefType* h = transposed_coefs_ + t_ * coefs_per_phase_;
			inputType* x_ptr = x - coefs_per_phase_ + 1;
			int offset = static_cast<int>(in - x_ptr);
			if (offset > 0)
			{
				// need to draw from the _state buffer
				inputType* state_ptr = state_end_ - offset;
				while (state_ptr < state_end_)
				{
					acc += *state_ptr++ * *h++;
				}
				x_ptr += offset;
			}
			while (x_ptr <= x)
			{
				acc += *x_ptr++ * *h++;
			}
			*y++ = acc;
			t_ += down_rate_;

			int advance_amount = t_ / up_rate_;

			x += advance_amount;
			// which phase of the filter to use
			t_ %= up_rate_;
		}
		x_offset_ = static_cast<int>(x - end);

		// manage _state buffer
		// find number of samples retained in buffer:
		int retain = (coefs_per_phase_ - 1) - in_count;
		if (retain > 0)
		{
			// for inCount smaller than state buffer, copy end of buffer
			// to beginning:
			copy(state_end_ - retain, state_end_, state_);
			// Then, copy the entire (short) input to end of buffer
			copy(in, end, state_end_ - in_count);
		}
		else
		{
			// just copy last input samples into state buffer
			copy(end - (coefs_per_phase_ - 1), end, state_);
		}
		// number of samples computed
		return static_cast<int>(y - out);
	}

	[[nodiscard]] int needed_out_count(const int in_count) const
	{
		const int np = in_count * up_rate_;
		int need = np / down_rate_;
		if ((t_ + up_rate_ * x_offset_) < (np % down_rate_))
			need++;
		return need;
	}

	[[nodiscard]] int coefs_per_phase() const { return coefs_per_phase_; }

private:
	int up_rate_;
	int down_rate_;

	coefType* transposed_coefs_;
	inputType* state_;
	inputType* state_end_;

	int padded_coef_count_; // ceil(len(coefs)/upRate)*upRate
	int coefs_per_phase_;   // _paddedCoefCount / upRate

	int t_; // "time" (modulo upRate)
	int x_offset_;
};




template <class S1, class S2, class C>
void upfirdn(int up_rate, int down_rate,
	S1* input, const int in_length, C* filter, int filter_length,
	vector<S2>& results)
	/*
	This template function provides a one-shot resampling.  Extra samples
	are padded to the end of the input in order to capture all the non-zero
	output samples.
	The output is in the "results" vector which is modified by the function.

	Note, I considered returning a vector instead of taking one on input, but
	then the C++ compiler has trouble with implicit template instantiation
	(e.g. have to say upfirdn<float, float, float> every time - this
	way we can let the compiler infer the template types).

	Thanks to Lewis Anderson (lkanders@ucsd.edu) at UCSD for
	the original version of this function.
	*/
{
	// Create the Resampler
	Resampler<S1, S2, C> the_resampler(up_rate, down_rate, filter, filter_length);

	// pad input by length of one poly phase of filter to flush all values out
	const int padding = the_resampler.coefs_per_phase() - 1;
	S1* input_padded = new S1[in_length + padding];
	for (int i = 0; i < in_length + padding; i++)
	{
		if (i < in_length)
			input_padded[i] = input[i];
		else
			input_padded[i] = 0;
	}

	// calc size of output
	int results_count = the_resampler.needed_out_count(in_length + padding);

	results.resize(results_count);

	// create filtering
	the_resampler.apply(input_padded,
		in_length + padding, &results[0], results_count);
	delete[] input_padded;
}

template <class S1, class S2, class C>
void upfirdn(const int up_rate, const int down_rate,
	vector<S1>& input, vector<C>& filter, vector<S2>& results)
	/*
	This template function provides a one-shot resampling.
	The output is in the "results" vector which is modified by the function.
	In this version, the input and filter are vectors as opposed to
	pointer/count pairs.
	*/
{
	upfirdn<S1, S2, C>(up_rate, down_rate, &input[0], static_cast<int>(input.size()), &filter[0],
		static_cast<int>(filter.size()), results);
}

//RESAMPLE  Change the sampling rate of a signal.
//   Y = RESAMPLE(UpFactor, DownFactor, InputSignal, OutputSignal) resamples the sequence in
//   vector InputSignal at UpFactor/DownFactor times and stores the resampled data to OutputSignal.
//   OutputSignal is UpFactor/DownFactor times the length of InputSignal. UpFactor and DownFactor must be
//   positive integers.

//This function is translated from Matlab's Resample function.

// Author: Haoqi Bai

template <typename T>
T sinc(T x)
{
	if (std::abs(x - 0.0) < 0.000001)
		return 1;

	const T pix = M_PI * x;
	return std::sin(pix) / pix;
}

inline int quotient_ceil(const int num1, const int num2)
{
	if (num1 % num2 != 0)
		return num1 / num2 + 1;
	return num1 / num2;
}

template <typename T>
std::vector<T> firls(int length, std::vector<T> freq, const std::vector<T>& amplitude)
{
	const int freq_size = static_cast<int>(freq.size());
	int weight_size = freq_size / 2;

	std::vector<T> weight(weight_size, 1.0);

	const int filter_length = length + 1;

	for (auto& it : freq)
		it /= 2.0;

	length = (filter_length - 1) / 2;
	const bool nodd = filter_length & 1;
	std::vector<T> k(length + 1);
	std::iota(k.begin(), k.end(), 0.0);
	if (!nodd)
		for (auto& it : k)
			it += 0.5;

	T b0 = 0.0;
	if (nodd)
		k.erase(k.begin());

	std::vector<T> b(k.size(), 0.0);
	for (int i = 0; i < freq_size; i += 2)
	{
		auto fi = freq[i];
		auto fip1 = freq[i + 1];
		auto ampi = amplitude[i];
		auto ampip1 = amplitude[i + 1];
		auto wt2 = std::pow(weight[i / 2], 2);
		auto m_s = (ampip1 - ampi) / (fip1 - fi);
		auto b1 = ampi - (m_s * fi);
		if (nodd)
			b0 += (b1 * (fip1 - fi)) + m_s / 2 * (std::pow(fip1, 2) - std::pow(fi, 2)) * wt2;

		const auto cosfib1 = std::cos(2 * M_PI * fip1);
		const auto cosfi = std::cos(2 * M_PI * fi);
		const auto num = m_s / (4 * M_PI * M_PI) * (cosfib1 - cosfi);

		std::transform(b.begin(), b.end(), k.begin(), b.begin(),
			[num, wt2](T b, T k) { return b + (num / (k * k)) * wt2; });


		std::transform(b.begin(), b.end(), k.begin(), b.begin(),
			[m_s, fi, fip1, wt2, b1](T b, T k)
			{ return b + (fip1 * (m_s * fip1 + b1) * sinc<T>(2 * k * fip1) -
				fi * (m_s * fi + b1) * sinc<T>(2 * k * fi)) *
			wt2; });
	}

	if (nodd)
	{
		b.insert(b.begin(), b0);
	}

	auto w0 = weight[0];
	std::vector<T> a(b.size());
	std::transform(b.begin(), b.end(),
		a.begin(),
		[w0](T b)
		{ return std::pow(w0, 2) * 4 * b; });

	std::vector<T> result = { a.rbegin(), a.rend() };
	decltype(a.begin()) it;
	if (nodd)
	{
		it = a.begin() + 1;
	}
	else
	{
		it = a.begin();
	}
	result.insert(result.end(), it, a.end());

	for (auto& it : result)
	{
		it *= 0.5;
	}

	return result;
}

template <typename T>
std::vector<T> kaiser(const int order, const T bta)
{
	T numerator, denominator;
	denominator = std::cyl_bessel_i(0, bta);
	const auto od2 = (static_cast<T>(order) - 1) / 2;
	std::vector<T> window(order);
	for (int n = 0; n < order; n++)
	{
		const auto nod = (n - od2) / od2;
		const auto nod2 = nod * nod;
		const auto x = bta * std::sqrt(1 - nod2);
		numerator = std::cyl_bessel_i(0, x);
		window[n] = numerator / denominator;
	}
	return window;
}

template <typename T>
std::vector<T> resample(int up_factor, int down_factor,
	std::vector<T>& input_signal)
{
	constexpr int n = 10;
	const T bta = 5.0;
	if (up_factor <= 0 || down_factor <= 0)
		throw std::runtime_error("factors must be positive integer");
	const int gcd_o = std::gcd(up_factor, down_factor);
	up_factor /= gcd_o;
	down_factor /= gcd_o;

	if (up_factor == down_factor)
	{
		return input_signal;
	}

	const int input_size = static_cast<int>(input_signal.size());

	int output_size = quotient_ceil(input_size * up_factor, down_factor);


	int max_factor = std::max(up_factor, down_factor);
	T firls_freq = 1.0 / 2.0 / static_cast<T>(max_factor);
	const int length = 2 * n * max_factor + 1;
	std::vector<T> firls_freqs_v = { 0.0, 2 * firls_freq, 2 * firls_freq, 1.0 };
	std::vector<T> firls_amplitude_v = { 1.0, 1.0, 0.0, 0.0 };
	std::vector<T> coefficients = firls<T>(length - 1, firls_freqs_v, firls_amplitude_v);
	std::vector<T> window = kaiser<T>(length, bta);
	for (size_t i = 0; i < coefficients.size(); i++)
		coefficients[i] *= up_factor * window[i];

	int length_half = (length - 1) / 2;
	int nz = down_factor - length_half % down_factor;
	std::vector<T> h;
	h.reserve(coefficients.size() + nz);
	for (int i = 0; i < nz; i++)
		h.push_back(0.0);
	for (size_t i = 0; i < coefficients.size(); i++)
		h.push_back(coefficients[i]);

	const int h_size = static_cast<int>(h.size());
	length_half += nz;
	const int delay = length_half / down_factor;
	nz = 0;
	while (quotient_ceil((input_size - 1) * up_factor + h_size + nz, down_factor) - delay < output_size)
		nz++;
	for (int i = 0; i < nz; i++)
		h.push_back(0.0);

	std::vector<T> y;
	upfirdn(up_factor, down_factor, input_signal, h, y);

	auto output_signal = std::vector<T>(output_size);

	for (int i = delay; i < output_size + delay; i++) {
		output_signal[i - delay] = y[i];
	}
	return output_signal;
}