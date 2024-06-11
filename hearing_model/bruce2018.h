#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ciso646>
#include <valarray>
#include <array>
#include <complex>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <random>

#include "resample.h"



enum Species
{
	CAT = 1,
	HUMAN_SHERA = 2,
	HUMAN_GLASSBERG_MOORE = 3
};

enum SynapseMappingFunction
{
	NONE = 0,
	SOFTPLUS = 1,
	EXPONENTIAL = 2,
	BOLTZMAN = 3
};

enum NoiseType
{
	ONES = 0,
	FIXED_MATLAB = 1,
	FIXED_SEED = 2,
	RANDOM = 3
};


enum PowerLaw
{
	APPROXIMATED = 0,
	ACTUAL = 1
};

namespace utils
{
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


	void fft(std::valarray<std::complex<double>>& x);
	void ifft(std::valarray<std::complex<double>>& x);
	extern std::mt19937 GENERATOR;

	std::vector<double> rand(size_t n);
	double rand1();
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

 
namespace pla
{
	std::vector<double> power_law(
		const std::vector<double>& amplitude_ihc, 
		NoiseType noise, 
		PowerLaw impl, 
		double spontaneous_firing_rate,
		double sampling_frequency, 
		double delay_point,
		double time_resolution,
		int n_total_timesteps
	);
}


namespace ihc
{

	//! Pass the signal through the Control path Third Order Nonlinear Gammatone Filte
	double delay_cat(double cf);

	/*
	 * @param px (pin) is the input sound wave in Pa sampled at the appropriate sampling rate (see instructions below)
	 * @param cf the characteristic frequency of the fiber in Hz
	 * @param nrep the number of repetitions for the psth
	 * @param tdres is the binsize in seconds, i.e., the reciprocal of the sampling rate (see instructions below)
	 * @param totalstim (int)floor(reptime/tdres+0.5)
	 * reptime is the time between stimulus repetitions in seconds - NOTE should be equal to or longer than the duration of pin
	 * @param cohc is the OHC scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
	 * @param cihc is the IHC scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
	 * @param species is the model species: "1" for cat, "2" for human with BM tuning from Shera et al. (PNAS 2002),
	 *    or "3" for human BM tuning from Glasberg & Moore (Hear. Res. 1990)
	 * @param ihcout
	 */

}


namespace syn
{
	/**
	 * model_Synapse_BEZ2018 - Bruce, Erfani & Zilany (2018) Auditory Nerve Model
	 *
	 *     psth = model_Synapse_BEZ2018(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel);
	 *
	 * For example,
	 *
	 *    psth = model_Synapse_BEZ2018(vihc,1e3,10,1/100e3,1,0,50,0.7,0.6); **requires 9 input arguments
	 *
	 * models a fiber with a CF of 1 kHz, a spontaneous rate of 50 /s, absolute and baseline relative
	 * refractory periods of 0.7 and 0.6 s, for 10 repetitions and a sampling rate of 100 kHz, and
	 * with variable fractional Gaussian noise and approximate implementation of the power-law functions
	 * in the synapse model.
	 *
	 * OPTIONAL OUTPUT VARIABLES:
	 *
	 *     [psth,meanrate,varrate,synout,trd_vector,trel_vector] = model_Synapse_BEZ2018(vihc,CF,nrep,dt,noiseType,implnt,spont,tabs,trel);
	 *
	 * NOTE ON SAMPLING RATE:-
	 * Since version 2 of the code, it is possible to run the model at a range
	 * of sampling rates between 100 kHz and 500 kHz.
	 * It is recommended to run the model at 100 kHz for CFs up to 20 kHz, and
	 * at 200 kHz for CFs> 20 kHz to 40 kHz.
	 *
	 * @param px (vihc) is the inner hair cell (IHC) relative transmembrane potential (in volts)
	 * @param cf the characteristic frequency of the fiber in Hz
	 * @param nrep the number of repetitions for the psth
	 * @param tdres the binsize in seconds, i.e., the reciprocal of the sampling rate (see instructions below)
	 * @param totalstim size of px
	 * @param noiseType is for "variable" or "fixed (frozen)" fGn: 1 for variable fGn and 0 for fixed (frozen) fGn
	 * @param implnt is for "approxiate" or "actual" implementation of the power-law functions: "0" for approx. and "1" for actual implementation
	 * @param spont the spontaneous firing rate in /s
	 * @param tabs  the absolute refractory period in /s
	 * @param trel the baselines mean relative refractory period in /s
	 * @param meanrate the analytical estimate of the mean firing rate in /s for each time bin
	 * @param varrate the analytical estimate of the variance in firing rate in /s for each time bin
	 * @param psth the peri-stimulus time histogram (PSTH) (or a spike train if nrep = 1)
	 * @param synout the synapse output rate in /s  for each time bin (before the effects of redocking are considered)
	 * @param trd_vector vector of the mean redocking time in s for each time bin
	 * @param trel_vector is a vector of the mean relative refractor period in s for each time bin
	 */

	//! Output wrapper for synapse
	struct SynapseOutput
	{
		int n_rep;
		int n_timesteps;
		int n_total_timesteps;

		std::vector<double> psth;
		std::vector<double> synaptic_output; // synout
		std::vector<double> redocking_time; // trd_vector

		// No pre-alloc
		std::vector<double> spike_times; 
		std::vector<double> mean_firing_rate; // meanrate
		std::vector<double> variance_firing_rate; // varrate
		std::vector<double> mean_relative_refractory_period; // trel_vector

		SynapseOutput(const int n_rep, const int n_timesteps) :
			n_rep(n_rep),
			n_timesteps(n_timesteps),
			n_total_timesteps(n_rep * n_timesteps),
			psth(n_timesteps),
			synaptic_output(n_total_timesteps),
			redocking_time(n_total_timesteps)
		{
		}
	};

	void up_sample_synaptic_output(const std::vector<double>& pla_out, double time_resolution, double sampling_frequency, int delay_point, SynapseOutput& res);

	template<size_t nSites>
	int spike_generator(
		double time_resolution,
		double spontaneous_firing_rate,
		double abs_refractory_period,
		double rel_refractory_period,
		SynapseOutput& res
	);
}


namespace stats
{
	void calculate_refractory_and_redocking_stats(
		syn::SynapseOutput& res,
		size_t n_rep,
		size_t n_sites,
		int totalstim,
		double absolute_refractory_period,
		double relative_refractory_period
	);
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
 * Since version 2 of the code, it is possible to run the model at a range
 * of sampling rates between 100 kHz and 500 kHz.
 * It is recommended to run the model at 100 kHz for CFs up to 20 kHz, and
 * at 200 kHz for CFs> 20 kHz to 40 kHz.
 *
 * @param sound_wave the input sound wave
 * @param cf characteristic frequency
 * @param n_rep the number of repetitions for the psth
 * @param time_resolution the binsize in seconds, i.e., the reciprocal of the sampling rate
 * @param rep_time the time duration of the competition
 * @param cohc is the OHC scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
 * @param cihc is the IHC scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
 * @param species is the model species: "1" for cat, "2" for human with BM tuning from Shera et al. (PNAS 2002),
 *    or "3" for human BM tuning from Glasberg & Moore (Hear. Res. 1990)
 * @returns
 */
std::vector<double> inner_hair_cell(
	const std::vector<double>& sound_wave,
	double cf = 1e3,
	int n_rep = 10,
	double time_resolution = 1 / 100e3, // binsize in seconds, recprocal of sampling rate
	double rep_time = 0.2, // repetition duration
	double cohc = 1,
	double cihc = 1,
	Species species = HUMAN_SHERA);

/**
 * Mapping Function from IHCOUT to input to the PLA
 * @param ihc_output 
 * @param spontaneous_firing_rate 
 * @param cf 
 * @param n_timesteps 
 * @param n_rep 
 * @param sampling_frequency 
 * @param time_resolution 
 * @param mapping_function 
 * @return 
 */
std::vector<double> map_to_synapse(
	const std::vector<double>& ihc_output,
	double spontaneous_firing_rate,
	double cf,
	int n_timesteps,
	int n_rep,
	double sampling_frequency,
	double time_resolution,
	SynapseMappingFunction mapping_function = SOFTPLUS // expliketype = 0
);

syn::SynapseOutput synapse(
	const std::vector<double>& amplitude_ihc, // px
	double cf,
	int n_rep,
	int n_timesteps,
	double time_resolution = 1 / 100e3, // time_resolution in seconds, recprocal of sampling rate
	NoiseType noise = RANDOM,
	PowerLaw pla_impl = APPROXIMATED, // implnttmp
	double spontaneous_firing_rate = 100,
	double abs_refractory_period = 0.7,
	double rel_refractory_period = 0.6,
	bool calculate_stats = true
);

