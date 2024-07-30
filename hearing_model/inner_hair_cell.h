#pragma once


#include <array>
#include <complex>
#include <vector>

#include "types.h"
#include "stimulus.h"

namespace ihc
{
	double delay_cat(double cf);

	struct MiddleEarFilter
	{
		double gain_max;
		std::array<std::array<double, 6>, 3> m;
		std::array<double, 3> mey1, mey2, mey3, px_cache;
		size_t i;

		MiddleEarFilter(double time_resolution, Species species);

		double operator()(double px);
	};

	struct WideBandGammaToneFilter
	{
		double time_resolution;
		double center_freq;
		int order;
		double cf;
		double delta_phase;
		double phase;
		std::array<std::complex<double>, 4> gtf;
		std::array<std::complex<double>, 4> gtf_last;

		double tau_wb_max;
		double tau_wb_min;
		double tau_wb;
		double bm_tau_min;
		double bm_tau_max;
		double tau_max;
		double tau_min;
		double ratio_bm;
		double wb_gain;

		double tmpcos;

		std::vector<double> gain_delay;
		double previous_gain;

		WideBandGammaToneFilter(
			double time_resolution,
			double cf,
			Species species,
			double cohc
		);

		double operator()(double me_out);

		//! shift of the location of poles of the C1 filter from the initial positions
		void shift_poles(double tau_c1, size_t n);

		[[nodiscard]] std::pair<double, int> calculate_gain_delay() const;
	};

	struct BoltzmanFilter
	{
		double asymptote;

		double operator()(double wb_out) const;
	};

	template <int Order>
	struct LowPassFilter
	{
		double c1_lp;
		double c2_lp;
		std::array<double, Order + 1> ohc;
		std::array<double, Order + 1> ohc_last;

		LowPassFilter(double time_resolution, double fc);

		double operator()(double x);
	};

	struct PostOhcFilter
	{
		double cohc;
		double bm_tau_min;
		double bm_tau_max;
		double min_r;
		double s0;

		PostOhcFilter(double cohc, double bm_tau_min, double bm_tau_max, double asym);

		double operator()(double x) const;
	};

	struct LogarithmicTransductionFunction
	{
		double slope;
		double asymptote;
		double strength;

		LogarithmicTransductionFunction(double slope, double asym);

		double operator()(double x) const;
	};

	struct ChirpFilter
	{
		double time_resolution;
		double cf;
		double tau_max;

		double init_phase;
		double gain_norm;

		std::array<std::array<double, 4>, 12> input;
		std::array<std::array<double, 4>, 12> output;
		std::array<std::complex<double>, 11> phase_array;

		double sigma0;
		double ipw;
		double ipb;
		double rpa;
		double r0;
		double fs_bi_linear;
		double gain_denominator;
		bool c1;

		ChirpFilter(double time_resolution, double cf_in, double bm_tau_max, bool c1 = true);

		void fill_phase_array(double r_sigma, bool multiply = false);

		double operator()(double me_out, double r_sigma);
	};
}


/**
 * @brief model_IHC_BEZ2018 - Bruce, Erfani & Zilany (2018) Auditory Nerve Model
 *
 *     vihc = model_IHC_BEZ2018(pin,CF,n_rep,dt,rep_time,cohc,cihc,species);
 *
 * vihc is the inner hair cell (IHC) relative transmembrane potential (in volts)

 * For example,
 *    vihc = model_IHC_BEZ2018(pin,1e3,10,1/100e3,0.2,1,1,2); **requires 8 input arguments
 *
 * models a normal human fiber of high spontaneous rate (normal OHC & IHC function) with a CF of 1 kHz,
 * for 10 repetitions and a sampling rate of 100 kHz, for a repetition duration of 200 ms, and
 * with approximate implementation of the power-law functions in the synapse model.
 *
 *
 * NOTE ON SAMPLING RATE:-
 * Since version 2 of the code, it is possible to create the model at a range
 * of sampling rates between 100 kHz and 500 kHz.
 * It is recommended to create the model at 100 kHz for CFs up to 20 kHz, and
 * at 200 kHz for CFs> 20 kHz to 40 kHz.
 *
 * @param stimulus the input sound wave
 * @param cf characteristic frequency
 * @param n_rep the number of repetitions for the psth
 * @param cohc is the OHC scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
 * @param cihc is the IHC scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
 * @param species is the model species: "1" for cat, "2" for human with BM tuning from Shera et al. (PNAS 2002),
 *    or "3" for human BM tuning from Glasberg & Moore (Hear. Res. 1990)
 * @returns 
 */
std::vector<double> inner_hair_cell(
	const stimulus::Stimulus& stimulus,
	double cf = 1e3,
	int n_rep = 10,
	double cohc = 1,
	double cihc = 1,
	Species species = HUMAN_SHERA
);
