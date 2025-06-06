# Bruce-Zilany-Carney Auditory Nerve Model (Python Interface)

This repository provides a modern Python interface to the auditory periphery model developed by the Bruce, Zilany, and Carney labs. The model simulates spike train responses in auditory nerve fibers with detailed physiological realism.

It is implemented in **C++ with `pybind11`**, offering performance and modularity. All dependencies on Matlab have been removed, making it fully standalone and suitable for Python-based environments. 

---

## 🧠 Model Background

This code is based on:

- **Bruce, I. C., Erfani, Y., & Zilany, M. S. A. (2018)**  
  _A phenomenological model of the synapse between the inner hair cell and auditory nerve: Implications of limited neurotransmitter release sites._  
  Hearing Research, 360:40–54.

- **Bruce, I., Buller, A., & Zilany, M. (2023)**  
  _Modeling of auditory nerve fiber input/output functions near threshold._  
  Acoustics 2023, Sydney, Australia.

> 📢 **Please cite both of the above if you publish research using this model or a modified version.**

---

## 🚀 Getting Started

### 1. Install prerequisites

You need a working C++14 compiler and Python 3.9+. Use a virtual environment if needed.

### 2. Clone and install

```bash
git clone https://github.com/jacobdenobel/brucezilany.git
cd brucezilany
pip install .
````

For development mode:

```bash
pip install -e .
```

---

## 🧪 Example Usage

```python
import numpy as np
from brucezilany import inner_hair_cell, synapse, stimulus

# Generate a stimulus
stim = stimulus.ramped_sine_wave(
    duration=0.05,
    simulation_duration=0.1,
    sampling_rate=100000,
    rt=0.005,
    delay=0.01,
    f0=1000,
    db=65
)

# IHC response
ihc_output = inner_hair_cell(stim, cf=1000)

# Synapse response
out = synapse(
    amplitude_ihc=ihc_output,
    cf=1000,
    n_rep=10,
    n_timesteps=stim.n_simulation_timesteps,
    spontaneous_firing_rate=100
)

print("Mean firing rate:", out.mean_firing_rate)
```

See `examples/` for more.

---

## ⚙️ Model Components

### `inner_hair_cell`

Simulates the receptor potential of an inner hair cell (IHC) in response to acoustic stimuli. Includes cochlear filtering and nonlinear transduction.

**Args**:

* `stimulus`: `Stimulus` object
* `cf`: characteristic frequency (Hz)
* `species`: `Species` enum
* `n_rep`: number of stimulus repetitions

Returns:

* IHC output as a vector (float)

---

### `synapse`

Simulates stochastic spike generation at the IHC-ANF synapse, using a 4-site vesicle model with adaptive redocking and realistic refractory behavior.

**Args**:

* `amplitude_ihc`: IHC potential vector
* `cf`: characteristic frequency (Hz)
* `n_rep`: number of stimulus repetitions
* `spontaneous_firing_rate`: in spikes/s
* `noise`: random noise mode (`NoiseType`)
* `pla_impl`: power-law adaptation method (`PowerLaw`)
* `rng`: optional seeded RNG

Returns:

* `SynapseOutput` object with PSTH, spike times, and statistics

---

## 🎧 Stimulus Handling

The `brucezilany.stimulus` module provides tools for generating and loading stimuli with precise control over sampling rate, duration, delay, and amplitude. This ensures accurate modeling of auditory nerve responses to various acoustic signals.

### 📦 Stimulus Class

Stimuli are represented as `Stimulus` objects that encapsulate both the waveform and simulation parameters.

```python
from brucezilany import stimulus

# Create a sine wave burst with a ramped onset/offset
stim = stimulus.ramped_sine_wave(
    duration=0.25,              # duration of tone burst (s)
    simulation_duration=0.3,    # total simulation time (s)
    sampling_rate=100_000,      # sampling rate (Hz)
    rt=2.5e-3,                  # rise/fall time (s)
    delay=25e-3,                # delay before tone onset (s)
    f0=5000,                    # frequency (Hz)
    db=60.0                     # SPL (dB)
)
```

You can inspect:

```python
print("Stimulus duration:", stim.stimulus_duration)
print("Sampling rate:", stim.sampling_rate)
print("Samples:", stim.data.shape)
```

---

### 📂 Load from File

You can also load `.wav` files (e.g., speech or natural sounds):

```python
stim = stimulus.from_file("data/defineit.wav", verbose=False)
```

By default, this:

* Resamples to 100 kHz
* Normalizes to 65 dB SPL
* Pads with silence to match a simulation time of 1s

---

### 📐 Normalization

If you want to adjust the dB level of a stimulus manually:

```python
stimulus.normalize_db(stim, stim_db=70)
```

## 🧰 Additional Tools

### `Neurogram` class

Multi-threaded simulation manager for full neurograms across CFs and fiber types.

```python
from brucecpp import Neurogram

ng = Neurogram(n_cf=50, n_low=5, n_med=10, n_high=35)
ng.create(sound_wave=stim.data, species="HUMAN_SHERA", n_rep=10)

output = ng.get_output()  # 3D array: [fiber, time, repetitions]
```

---

## 🛠 Refactoring & Enhancements

Compared to the original code, the following major changes were made:

* ✅ **All Matlab dependencies removed**

  * C++ replacements for `resample`, `randn`, etc.
  * Fully standalone backend
* ✅ **Modern C++14 style**

  * No `new/delete`, safer memory via STL containers
  * Cleaner APIs and encapsulation
* ✅ **Modular structure**

  * Clear split between IHC, synapse, stimulus, and neurogram
* ✅ **Python bindings via pybind11**

  * Fully exposed enums, structured output objects
* ✅ **Multi-threading support**

  * `Neurogram` uses parallel processing for simulating many CF/fiber combinations

The original Matlab code can be found [here](https://www.ece.mcmaster.ca/~ibruce/zbcANmodel/zbcANmodel.htm). A reference to the code use to build this repository is included as a .zip file in the data/ folder.

---

## 🧠 References

Additional foundational papers:

* Zilany, M. S. A., Bruce, I. C., & Carney, L. H. (2014).
  *Updated parameters and expanded simulation options for a model of the auditory periphery.*
  JASA, 135(1):283–286.

* Zilany, M. S. A., Bruce, I. C., Nelson, P. C., & Carney, L. H. (2009).
  *A phenomenological model of the synapse between the inner hair cell and auditory nerve: Long-term adaptation with power-law dynamics.*
  JASA, 126(5):2390–2412.

---

## 📬 Contact

Questions or contributions? Open a GitHub issue or pull request.


