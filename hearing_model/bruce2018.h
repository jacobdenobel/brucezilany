#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ciso646>
#define _USE_MATH_DEFINES
#include <cmath>
#include <valarray>
#include <complex>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <random>

#include "resample.h"

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

enum Species {
    CAT = 1, HUMAN_SHERA = 2, HUMAN_GLASSBERG_MOORE = 3
};

/**
 * @brief model_IHC_BEZ2018 - Bruce, Erfani & Zilany (2018) Auditory Nerve Model
 *
 *     vihc = model_IHC_BEZ2018(pin,CF,nrep,dt,reptime,cohc,cihc,species);
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
 * @param nrep the number of repetitions for the psth
 * @param binsize the binsize in seconds, i.e., the reciprocal of the sampling rate
 * @param cohc is the OHC scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
 * @param cihc is the IHC scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
 * @param species is the model species: "1" for cat, "2" for human with BM tuning from Shera et al. (PNAS 2002),
 *    or "3" for human BM tuning from Glasberg & Moore (Hear. Res. 1990)
 * @returns 
 */
std::vector<double> inner_hair_cell(
    const std::vector<double> sound_wave,
    const double cf = 1e3,
    const size_t nrep = 10,
    const double binsize = 1 / 100e3, // binsize in seconds, recprocal of sampling rate
    const double reptime = 0.2,       // repetition duration
    const double cohc = 1,
    const double cihc = 1,
    const Species species = HUMAN_SHERA);

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
void IHCAN(double* px, double cf, int nrep, double tdres, int totalstim,
    double cohc, double cihc, int species, double* ihcout);

double C1ChirpFilt(double, double, double, int, double, double);
double C2ChirpFilt(double, double, double, int, double, double);
double WbGammaTone(double, double, double, int, double, double, int);
double Get_tauwb(double, int, int, double*, double*);
double Get_taubm(double, int, double, double*, double*, double*);
double gain_groupdelay(double, double, double, double, int*);
double delay_cat(double cf);
double delay_human(double cf);
double OhcLowPass(double, double, double, int, double, int);
double IhcLowPass(double, double, double, int, double, int);
double Boltzman(double, double, double, double, double);
double NLafterohc(double, double, double, double);
double NLogarithm(double, double, double, double);

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
void SingleAN(const std::vector<double>& sampIHC, double cf, int nrep, double tdres, int totalstim, double noiseType, double implnt, double spont, double tabs, double trel, double* meanrate, double* varrate, double* psth, double* synout, double* trd_vector, double* trel_vector);
double Synapse(const std::vector<double>&, double, double, int, int, double, double, double, double, double*);
int SpikeGenerator(double*, double, double, double, double, double, int, double, double,
    double, int, int, double, std::vector<double>&, double*);

#define MAXSPIKES 1000000
#ifndef TWOPI
#define TWOPI 6.28318530717959
#endif


template<typename T>
inline void validate_parameter(const T p, const T lb, const T ub, const std::string& name = "") {
    if (p < lb or p > ub) {
        std::ostringstream ss;
        ss << name << "= " << p << " is out of bounds [" << lb << ", " << ub << "]";
        std::cout << ss.str() << std::endl;
        throw std::invalid_argument(ss.str());
    }
}


void fft(CArray& x);
void ifft(CArray& x);
extern std::mt19937 GENERATOR;

std::vector<double> rand(const size_t n);
double rand1();
std::vector<double> randn(const size_t n);
std::vector<double> ffgn(int N = 5300, const double tdres = 1e-4, const double Hinput = 0.9, const int noiseType = 1, const int mu = 100);


//! Output wrapper for synapse
struct SynapseOutput {
    std::vector<double> psth;
    std::vector<double> mean_firing_rate;                // meanrate
    std::vector<double> variance_firing_rate;            // varrate
    std::vector<double> output_rate;                     // synout
    std::vector<double> mean_redocking_time;             // trd_vector
    std::vector<double> mean_relative_refractory_period; // trel_vector

    SynapseOutput(const size_t n, const size_t n2): psth(n), mean_firing_rate(n), variance_firing_rate(n),
        output_rate(n2), mean_redocking_time(n2), mean_relative_refractory_period(n2) {}
};

SynapseOutput synapse(
    std::vector<double>& pressure_wave, // px
    const double cf,
    const size_t nrep,
    const int totalstim,
    const double binsize = 1 / 100e3, // binsize in seconds, recprocal of sampling rate
    const bool noise = false,    
    const bool approximate = true,       // implnttmp
    const double spont_rate = 100,     
    const double abs_refractory_period = 0.7,
    const double rel_refractory_period = 0.6
);


std::vector<double> map_to_power_law(double* ihcout, double spont, double cf, int totalstim, int nrep, double sampFreq, double tdres);