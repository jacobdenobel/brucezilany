/* This is the BEZ2018 version of the code for auditory periphery model from the Carney, Bruce and Zilany labs.
 *
 * This release implements the version of the model described in:
 *
 *   Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
 *   model of the synapse between the inner hair cell and auditory nerve:
 *   Implications of limited neurotransmitter release sites," to appear in
 *   Hearing Research. (Special Issue on "Computational Models in Hearing".)
 *
 * Please cite this paper if you publish any research
 * results obtained with this code or any modified versions of this code.
 *
 * See the file readme.txt for details of compiling and running the model.
 *
 * %%% Ian C. Bruce (ibruce@ieee.org), Yousof Erfani (erfani.imag()ousof@gmail.com),
 *     Muhammad S. A. Zilany (msazilany@gmail.com) - December 2017 %%%
 *
 */

#define _USE_MATH_DEFINES
#include <cmath>

#include "inner_hair_cell.h"
#include "utils.h"

static constexpr double TWO_PI = 2.0 * M_PI;

namespace
{
	double get_q10(const double cf, const Species species)
	{
		switch (species)
		{
		case CAT:
			return pow(10, 0.4708 * log10(cf / 1e3) + 0.4664);
		case HUMAN_SHERA:
			return pow((cf / 1000), 0.3) * 12.7 * 0.505 + 0.2085;
		case HUMAN_GLASSBERG_MOORE:
			[[fallthrough]];
		default:
			return cf / 24.7 / (4.37 * (cf / 1000) + 1) * 0.505 + 0.2085;
		}
	}

	double get_center_frequency(const double cf, const Species species)
	{
		/** Calculate the center frequency for the control-path wideband filter
			from the location on basilar membrane, based on Greenwood (JASA 1990) */
		if (species == CAT) /* for cat */
		{
			/* Cat frequency shift corresponding to 1.2 mm */
			const double bm_place = 11.9 * log10(0.80 + cf / 456.0);
			/* Calculate the location on basilar membrane from CF */
			return 456.0 * (pow(10, (bm_place + 1.2) / 11.9) - 0.80); /* shift the center freq */
		}

		/* Human frequency shift corresponding to 1.2 mm */
		const double bm_place = (35 / 2.1) * log10(1.0 + cf / 165.4);
		/* Calculate the location on basilar membrane from CF */
		return 165.4 * (pow(10, (bm_place + 1.2) / (35 / 2.1)) - 1.0); /* shift the center freq */
	}
}

namespace ihc
{
	/* -------------------------------------------------------------------------------------------- */
	/** Calculate the delay (basilar membrane, synapse, etc. for cat) */
	double delay_cat(const double cf)
	{
		constexpr double a0 = 3.0;
		constexpr double a1 = 12.5;
		const double x = 11.9 * log10(0.80 + cf / 456.0); /* cat mapping */
		const double delay = a0 * exp(-x / a1) * 1e-3;

		return delay;
	}


	MiddleEarFilter::MiddleEarFilter(
		const double time_resolution,
		const Species species
	): m{}, mey1{}, mey2{}, mey3{}, px_cache{}, i{0}
	{
		const double c = (TWO_PI * 1e3 / tan(TWO_PI / 2 * 1e3 * time_resolution));
		if (species == CAT) /* for cat */
		{
			/* Cat middle-ear filter - simplified version from Bruce et al. (JASA 2003) */
			m[0][0] = c / (c + 693.48);
			m[0][1] = (693.48 - c) / c;
			m[0][2] = 0.0;
			m[0][3] = 1.0;
			m[0][4] = -1.0;
			m[0][5] = 0.0;
			m[1][0] = 1 / (pow(c, 2) + 11053 * c + 1.163e8);
			m[1][1] = -2 * pow(c, 2) + 2.326e8;
			m[1][2] = pow(c, 2) - 11053 * c + 1.163e8;
			m[1][3] = pow(c, 2) + 1356.3 * c + 7.4417e8;
			m[1][4] = -2 * pow(c, 2) + 14.8834e8;
			m[1][5] = pow(c, 2) - 1356.3 * c + 7.4417e8;
			m[2][0] = 1 / (pow(c, 2) + 4620 * c + 909059944);
			m[2][1] = -2 * pow(c, 2) + 2 * 909059944;
			m[2][2] = pow(c, 2) - 4620 * c + 909059944;
			m[2][3] = 5.7585e5 * c + 7.1665e7;
			m[2][4] = 14.333e7;
			m[2][5] = 7.1665e7 - 5.7585e5 * c;
			gain_max = 41.1405;
		}
		else /* for human */
		{
			/* Human middle-ear filter - based on Pascal et al. (JASA 1998)  */
			m[0][0] = 1 / (pow(c, 2) + 5.9761e+003 * c + 2.5255e+007);
			m[0][1] = (-2 * pow(c, 2) + 2 * 2.5255e+007);
			m[0][2] = (pow(c, 2) - 5.9761e+003 * c + 2.5255e+007);
			m[0][3] = (pow(c, 2) + 5.6665e+003 * c);
			m[0][4] = -2 * pow(c, 2);
			m[0][5] = (pow(c, 2) - 5.6665e+003 * c);
			m[1][0] = 1 / (pow(c, 2) + 6.4255e+003 * c + 1.3975e+008);
			m[1][1] = (-2 * pow(c, 2) + 2 * 1.3975e+008);
			m[1][2] = (pow(c, 2) - 6.4255e+003 * c + 1.3975e+008);
			m[1][3] = (pow(c, 2) + 5.8934e+003 * c + 1.7926e+008);
			m[1][4] = (-2 * pow(c, 2) + 2 * 1.7926e+008);
			m[1][5] = (pow(c, 2) - 5.8934e+003 * c + 1.7926e+008);
			m[2][0] = 1 / (pow(c, 2) + 2.4891e+004 * c + 1.2700e+009);
			m[2][1] = (-2 * pow(c, 2) + 2 * 1.2700e+009);
			m[2][2] = (pow(c, 2) - 2.4891e+004 * c + 1.2700e+009);
			m[2][3] = (3.1137e+003 * c + 6.9768e+008);
			m[2][4] = 2 * 6.9768e+008;
			m[2][5] = (-3.1137e+003 * c + 6.9768e+008);
			gain_max = 2;
		}
	}

	double MiddleEarFilter::operator()(const double px)
	{
		const size_t i0 = i % 3;
		const size_t i1 = (i - 1) % 3;
		const size_t i2 = (i++ - 2) % 3;
		px_cache[i0] = px;

		mey1[i0] = m[0][0] * (-m[0][1] * mey1[i1] - m[0][2] * mey1[i2] + m[0][3] * px_cache[i0] + m[0][4] * px_cache
			[i1] + m[0][5] * px_cache[i2]);
		mey2[i0] = m[1][0] * (-m[1][1] * mey2[i1] - m[1][2] * mey2[i2] + m[1][3] * mey1[i0] + m[1][4] * mey1[i1] + m
			[1][5] * mey1[i2]);
		mey3[i0] = m[2][0] * (-m[2][1] * mey3[i1] - m[2][2] * mey3[i2] + m[2][3] * mey2[i0] + m[2][4] * mey2[i1] + m
			[2][5] * mey2[i2]);
		return mey3[i0] / gain_max;
	}


	WideBandGammaToneFilter::WideBandGammaToneFilter(
		const double time_resolution,
		const double cf,
		const Species species,
		const double cohc
	)
		:
		time_resolution(time_resolution),
		center_freq(get_center_frequency(cf, species)),
		order(3),
		cf(cf),
		delta_phase(-TWO_PI * center_freq * time_resolution),
		phase(0.0),
		previous_gain(0.0)
	{
		const double gain = std::max(std::min(52.0 / 2.0 * (tanh(2.2 * log10(cf / 0.6e3) + 0.15) + 1.0), 60.0), 15.0);
		const double ratio = pow(10, (-gain / (20.0 * order)));
		const double bw = cf / get_q10(cf, species);
		tau_max = 2.0 / (TWO_PI * bw);
		tau_min = tau_max * ratio;

		tau_wb_max = tau_min + 0.2 * (tau_max - tau_min);
		tau_wb_min = tau_wb_max / tau_max * tau_min;

		constexpr double bw_factor = 0.7;
		constexpr double factor = 2.5;
		ratio_bm = pow(10, (-gain / (20.0 * factor)));
		bm_tau_max = tau_max / bw_factor;
		bm_tau_min = bm_tau_max * ratio_bm;

		tmpcos = cos(TWO_PI * (center_freq - cf) * time_resolution);
		gain_delay.resize(size_t{ 1 } + static_cast<size_t>(floor(1.0 / (2.0 * std::sqrt(1 - pow(tmpcos, 2))))));

		shift_poles(cohc * (bm_tau_max - bm_tau_min) + bm_tau_min, 0);
		wb_gain = previous_gain;
	}

	double WideBandGammaToneFilter::operator()(const double me_out)
	{
		phase += delta_phase;

		const double dt = tau_wb * 2.0 / time_resolution;
		const double c1_lp = (dt - 1) / (dt + 1);
		const double c2_lp = 1.0 / (dt + 1);

		gtf[0] = std::complex<double>(cos(phase), sin(phase)) * me_out;

		/* IIR Bi-linear transformation LPF */
		for (size_t j = 1; j <= static_cast<size_t>(order); j++)
			gtf[j] = ((c2_lp * wb_gain) * (gtf[j - 1] + gtf_last[j - 1])) + (c1_lp * gtf_last[j]);

		const double out = (cos(-phase) * gtf[order].real()) - (sin(-phase) * gtf[order].imag());

		for (size_t i = 0; i <= static_cast<size_t>(order); i++)
			gtf_last[i] = gtf[i];

		return pow((tau_wb / tau_wb_max), order) * out * 10e3 * std::max(1., cf / 5e3);
	}

	//! shift of the location of poles of the C1 filter from the initial positions
	void WideBandGammaToneFilter::shift_poles(const double tau_c1, const size_t n)
	{
		tau_wb = tau_wb_max + (tau_c1 - bm_tau_max) * (tau_wb_max - tau_wb_min) / (bm_tau_max - bm_tau_min);

		const auto [current_gain, current_gain_delay] = calculate_gain_delay();

		gain_delay[(n + current_gain_delay) % gain_delay.size()] = previous_gain;

		// TODO: can we optimize this?
		if (gain_delay[n % gain_delay.size()] == 0)
			gain_delay[n % gain_delay.size()] = previous_gain;

		previous_gain = current_gain;

		wb_gain = gain_delay[n % gain_delay.size()];
		gain_delay[n % gain_delay.size()] = 0.0;
	}

	[[nodiscard]] std::pair<double, int> WideBandGammaToneFilter::calculate_gain_delay() const
	{
		const double dt = tau_wb * 2.0 / time_resolution;
		const double c1_lp = (dt - 1) / (dt + 1);
		const double c2_lp = 1.0 / (dt + 1);
		const double tmp1 = 1 + c1_lp * c1_lp - 2 * c1_lp * tmpcos;
		const double tmp2 = 2 * c2_lp * c2_lp * (1 + tmpcos);
		const int current_gain_delay = static_cast<int>(floor(
			0.5 - (c1_lp * c1_lp - c1_lp * tmpcos) / (1 + c1_lp * c1_lp - 2 * c1_lp * tmpcos)));
		const double current_gain = pow(tmp1 / tmp2, 1.0 / 2.0);
		return {current_gain, current_gain_delay};
	}


	double BoltzmanFilter::operator()(const double wb_out) const
	{
		constexpr double s0 = 12.0;
		constexpr double s1 = 5.0;
		constexpr double x1 = 5.0;


		const double shift = 1.0 / (1.0 + asymptote);
		const double x0 = s0 * log((1.0 / shift - 1) / (1 + exp(x1 / s1)));
		const double out1 = 1.0 / (1.0 + exp(-(wb_out - x0) / s0) * (1.0 + exp(-(wb_out - x1) / s1))) - shift;
		const double out = out1 / (1 - shift);
		return out;
	}


	template <int Order>
	LowPassFilter<Order>::LowPassFilter(const double time_resolution, const double fc): ohc{}, ohc_last{}
	{
		const double c = 2.0 / time_resolution;
		c1_lp = (c - TWO_PI * fc) / (c + TWO_PI * fc);
		c2_lp = TWO_PI * fc / (TWO_PI * fc + c);
	}

	template <int Order>
	double LowPassFilter<Order>::operator()(const double x)
	{
		constexpr double gain = 1.0;

		ohc[0] = x * gain;
		for (int i = 0; i < Order; i++)
			ohc[i + 1] = c1_lp * ohc_last[i + 1] + c2_lp * (ohc[i] + ohc_last[i]);

		for (int j = 0; j <= Order; j++)
			ohc_last[j] = ohc[j];

		return ohc[Order];
	}


	PostOhcFilter::PostOhcFilter(const double cohc, const double bm_tau_min, const double bm_tau_max, const double asym)
		: cohc(cohc),
		  bm_tau_min(bm_tau_min),
		  bm_tau_max(bm_tau_max),
		  min_r(0.05)
	{
		const double r = bm_tau_min / bm_tau_max;
		if (r < min_r)
			min_r = 0.5 * r;

		const double dc = (asym - 1) / (asym + 1.0) / 2.0 - min_r;
		const double r1 = r - min_r;
		s0 = -dc / log(r1 / (1 - min_r));
	}

	double PostOhcFilter::operator()(const double x) const
	{
		const double x1 = fabs(x);
		double out = std::min(std::max(bm_tau_max * (min_r + (1.0 - min_r) * exp(-x1 / s0)), bm_tau_min),
		                      bm_tau_max);

		out = cohc * (out - bm_tau_min) + bm_tau_min;

		if (1 / out < 0.0)
			printf("The poles are in the right-half plane; system is unstable.\n");
		return out;
	}


	LogarithmicTransductionFunction::LogarithmicTransductionFunction(const double slope, const double asym):
		slope(slope),
		asymptote(asym),
		strength(20.0e6 / pow(10, 80.0 / 20))
	{
	}

	double LogarithmicTransductionFunction::operator()(const double x) const
	{
		const double xx = log(1.0 + strength * fabs(x)) * slope;

		if (x < 0)
		{
			const double splx = 20 * log10(-x / 20e-6);
			const double asym_t = asymptote - (asymptote - 1) / (1 + exp(splx / 5.0));
			return -1 / asym_t * xx;
		}
		return xx;
	}


	ChirpFilter::ChirpFilter(const double time_resolution, const double cf_in, const double bm_tau_max, const bool c1) :
		time_resolution(time_resolution),
		cf(cf_in * TWO_PI),
		tau_max(bm_tau_max),
		init_phase(0.0),
		gain_norm(1.0),
		input{},
		output{},
		phase_array{},
		sigma0{1 / bm_tau_max},
		ipw{1.01 * cf - 50},
		ipb{0.2343 * cf - 1104},
		rpa{pow(10, log10(cf_in) * 0.9 + 0.55) + 2000},
		r0{-(pow(10, log10(cf_in) * 0.7 + 1.6) + 500)},
		fs_bi_linear{cf / tan(cf * time_resolution / 2)},
		gain_denominator{pow(sqrt(cf * cf + r0 * r0), 5)},
		c1(c1)

	{
		fill_phase_array(0.0);

		for (size_t i = 1; i <= size_t{ 5 }; i++)
		{
			const auto& p = phase_array[i * 2 - 1];
			init_phase += atan(cf / -r0) - atan((cf - p.imag()) / -p.real()) - atan((cf + p.imag()) / -p.real());
		}

		for (int r = 1; r <= 10; r++)
			gain_norm *= pow((cf - phase_array[r].imag()), 2) + phase_array[r].real() * phase_array[r].real();
	}

	void ChirpFilter::fill_phase_array(const double r_sigma, const bool multiply)
	{
		phase_array[1].real(-sigma0 - r_sigma);
		if (multiply)
			phase_array[1].real(-sigma0 * r_sigma);

		if (phase_array[1].real() > 0.0)
			printf("The system becomes unstable.\n");

		phase_array[1].imag(ipw);

		phase_array[5].real(phase_array[1].real() - rpa);
		phase_array[5].imag(phase_array[1].imag() - ipb);

		phase_array[3].real((phase_array[1].real() + phase_array[5].real()) * 0.5);
		phase_array[3].imag((phase_array[1].imag() + phase_array[5].imag()) * 0.5);

		phase_array[2] = std::conj(phase_array[1]);
		phase_array[4] = std::conj(phase_array[3]);
		phase_array[6] = std::conj(phase_array[5]);

		phase_array[7] = phase_array[1];
		phase_array[8] = phase_array[2];
		phase_array[9] = phase_array[5];
		phase_array[10] = phase_array[6];
	}

	double ChirpFilter::operator()(const double me_out, const double r_sigma)
	{
		constexpr int order_of_zero = 5;
		constexpr size_t half_order_pole = 5;


		const double norm_gain = sqrt(gain_norm) / gain_denominator;

		fill_phase_array(r_sigma, !c1);

		double phase = 0.0;
		for (size_t i = 1; i <= half_order_pole; i++)
		{
			const auto& p = phase_array[i * 2 - 1];
			phase = phase - atan((cf - p.imag()) / -p.real()) - atan((cf + p.imag()) / -p.real());
		}

		r0 = -cf / tan((init_phase - phase) / order_of_zero);

		if (r0 > 0.0)
			printf("The zeros are in the right-half plane.\n");

		input[1][3] = input[1][2];
		input[1][2] = input[1][1];
		input[1][1] = me_out;

		double dy;
		for (size_t i = 1; i <= half_order_pole; i++)
		{
			const auto& p = phase_array[i * 2 - 1];

			const double temp = pow((fs_bi_linear - p.real()), 2) + pow(p.imag(), 2);
			dy = input[i][1] * (fs_bi_linear - r0) - 2 * r0 * input[i][2] - (fs_bi_linear + r0) * input[i][3] + 2 *
				output[i][1] * (fs_bi_linear * fs_bi_linear - p.real() * p.real() - p.imag() * p.imag()) - output[i]
				[2] * ((fs_bi_linear + p.real()) * (fs_bi_linear + p.real()) + p.imag() * p.imag());

			dy = dy / temp;


			input[i + 1][3] = output[i][2];
			input[i + 1][2] = output[i][1];
			input[i + 1][1] = dy;

			output[i][2] = output[i][1];
			output[i][1] = dy;
		}

		dy = output[half_order_pole][1] * norm_gain; /* don't forget the gain term */
		return dy / 4.0; /* signal path output is divided by 4 to give correct C1 filter gain */
	}
}


std::vector<double> inner_hair_cell(
	const stimulus::Stimulus& stimulus,
	const double cf,
	const int n_rep,
	const double cohc,
	const double cihc,
	const Species species)
{
	if (species == CAT)
		utils::validate_parameter(cf, 124.9, 40.1e3, "cf");
	else
		utils::validate_parameter(cf, 124.9, 20.1e3, "cf");

	utils::validate_parameter(n_rep, 1, std::numeric_limits<int>::max(), "n_rep");
	utils::validate_parameter(stimulus.simulation_duration, stimulus.stimulus_duration,
	                          std::numeric_limits<double>::infinity(), "stimulus.simulation_duration");
	utils::validate_parameter(cohc, 0., 1., "cohc");
	utils::validate_parameter(cihc, 0., 1., "cihc");

	std::vector<double> output(stimulus.n_simulation_timesteps * n_rep);

	ihc::MiddleEarFilter me_filter(stimulus.time_resolution, species);
	ihc::WideBandGammaToneFilter wb_filter(stimulus.time_resolution, cf, species, cohc);
	ihc::BoltzmanFilter boltzman_filter{7.0};
	ihc::LowPassFilter<2> ohc_low_pass_filter(stimulus.time_resolution, 600.);
	ihc::PostOhcFilter non_linear_after_ohc_filter{cohc, wb_filter.bm_tau_min, wb_filter.bm_tau_max, 7.0};
	ihc::ChirpFilter c1_chirp_filter{ stimulus.time_resolution, cf, wb_filter.bm_tau_max, true};
	ihc::ChirpFilter c2_chirp_filter{ stimulus.time_resolution, cf, wb_filter.bm_tau_max, false};
	ihc::LogarithmicTransductionFunction ltf_1{0.1, 3.0};
	ihc::LogarithmicTransductionFunction ltf_2{0.2, 1.0};
	ihc::LowPassFilter<7> ihc_low_pass_filter(stimulus.time_resolution, 3000);

	const double delay = ihc::delay_cat(cf); // human uses same delay function
	const int delay_point = std::max(0, static_cast<int>(ceil(delay / stimulus.time_resolution)));

	for (size_t n = 0; n < stimulus.n_simulation_timesteps; n++)
	{
		const double px = n < stimulus.n_stimulation_timesteps ? stimulus.data[n] : 0.0;
		const double me_out = me_filter(px);
		const double wb_out = wb_filter(me_out);
		const double ohc_nonlinear_out = boltzman_filter(wb_out);
		const double ohc_out = ohc_low_pass_filter(ohc_nonlinear_out);
		const double tau_c1 = non_linear_after_ohc_filter(ohc_out);

		wb_filter.shift_poles(tau_c1, n);

		const double c1_filter_out = c1_chirp_filter(me_out, 1 / tau_c1 - 1 / wb_filter.bm_tau_max);
		const double c2_filter_out = c2_chirp_filter(me_out, 1 / wb_filter.ratio_bm);

		const double c1_ihc = ltf_1(cihc * c1_filter_out);
		const double c2_ihc = -ltf_2(c2_filter_out * fabs(c2_filter_out) * cf / 10 * cf / 2e3);

		const double ihc_out = ihc_low_pass_filter(c1_ihc + c2_ihc);

		if (n + delay_point < stimulus.n_simulation_timesteps)
			for (int j = 0; j < n_rep; j++)
				output[(stimulus.n_simulation_timesteps * j) + n + delay_point] = ihc_out;
	}
	return output;
}
