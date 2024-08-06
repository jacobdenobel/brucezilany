#pragma once

enum Species
{
	CAT = 1,
	HUMAN_SHERA = 2,
	HUMAN_GLASSBERG_MOORE = 3
};

enum SynapseMapping
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