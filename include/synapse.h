#pragma once

#include <optional>
#include <vector>
#include "utils.h"

namespace syn
{
	//! Output wrapper for synapse model
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
		/*
		 * @param meanrate the analytical estimate of the mean firing rate in /s for each time bin
		 * @param varrate the analytical estimate of the variance in firing rate in /s for each time bin
		 * @param psth the peri-stimulus time histogram (PSTH) (or a spike train if nrep = 1)
		 * @param synout the synapse output rate in /s  for each time bin (before the effects of redocking are considered)
		 * @param trd_vector vector of the mean redocking time in s for each time bin
		 * @param trel_vector is a vector of the mean relative refractor period in s for each time bin
		 */
	};

	/**
	 * Up sample the result of the power law function to the original (High 100 kHz) sampling rate
	 *
	 * @param pla_out The output of the power law mapping
	 * @param time_resolution the time resolution of the model
	 * @param sampling_frequency the required sampling frequency
	 * @param delay_point the delay point of the model
	 * @param res the up sampled signal
	 */
	void up_sample_synaptic_output(const std::vector<double>& pla_out, double time_resolution, double sampling_frequency, int delay_point, SynapseOutput& res);

	/**
	 * The spike generator model.
	 *
	 * @tparam nSites The number of adaptive re-docking sites
	 * @param time_resolution The time resolution of the model
	 * @param spontaneous_firing_rate The spontaneous firing rate
	 * @param abs_refractory_period The absolute refractory period
	 * @param rel_refractory_period the relative refractory period 
	 * @param res The output container
	 */
	template<size_t nSites>
	int spike_generator(
		double time_resolution,
		double spontaneous_firing_rate,
		double abs_refractory_period,
		double rel_refractory_period,
		SynapseOutput& res
	);

	/**
	 *	Calculate (optional) extended statistics
	 *
	 * @param n_sites The number of adaptive re-docking sites
	 * @param abs_refractory_period The absolute refractory period
	 * @param rel_refractory_period the relative refractory period
	 * @param res The output container
	 */
	void calculate_refractory_and_redocking_stats(
		int n_sites,
		double abs_refractory_period,
		double rel_refractory_period,
		SynapseOutput& res
	);
}


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
 * Since version 2 of the code, it is possible to create the model at a range
 * of sampling rates between 100 kHz and 500 kHz.
 * It is recommended to create the model at 100 kHz for CFs up to 20 kHz, and
 * at 200 kHz for CFs> 20 kHz to 40 kHz.
 *
 * @param amplitude_ihc (vihc) is the inner hair cell (IHC) relative transmembrane potential (in volts)
 * @param cf the characteristic frequency of the fiber in Hz
 * @param n_rep the number of repetitions for the psth
 * @param n_timesteps the number of timesteps
 * @param time_resolution the binsize in seconds, i.e., the reciprocal of the sampling rate (see instructions below)
 * @param noise The type of noise (see `NoiseType') is for "variable" or "fixed (frozen)" fGn: 1 for variable fGn and 0 for fixed (frozen) fGn
 * @param pla_impl The type of power law implementation is for "approxiate" or "actual" implementation of the power-law functions: "0" for approx. and "1" for actual implementation
 * @param spontaneous_firing_rate the spontaneous firing rate in /s
 * @param abs_refractory_period  the absolute refractory period in /s
 * @param rel_refractory_period the baselines mean relative refractory period in /s
 * @param calculate_stats Whether to calculate optional statistics
 * @param rng Random generator object. If nullptr, global SEED will be used to instantiate
 */
syn::SynapseOutput synapse(
	const std::vector<double>& amplitude_ihc, // px
	double cf,
	int n_rep,
	size_t n_timesteps,
	double time_resolution = 1 / 100e3, // time_resolution in seconds, reciprocal of sampling rate
	NoiseType noise = RANDOM,
	PowerLaw pla_impl = APPROXIMATED, 
	double spontaneous_firing_rate = 100,
	double abs_refractory_period = 0.7,
	double rel_refractory_period = 0.6,
	bool calculate_stats = true,
	std::optional<utils::RandomGenerator> rng = std::nullopt
);