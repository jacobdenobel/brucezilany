# Bruce-Zilany-Carney Auditory Nerve Model (Python Interface)

This repository provides a modern Python interface to the auditory periphery model developed by the Bruce, Zilany, and Carney labs. The model simulates spike train responses in auditory nerve fibers with detailed physiological realism.

It is implemented in **C++ with `pybind11`**. All dependencies on Matlab have been removed, making it fully standalone and suitable for Python-based environments. In general, the library should have feature-parity with the original source, but the code should be *considerably faster* to execute. 

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

### Install from PyPi

Alternatively, you can also install the package from pip, if you want to include it as dependency.

```bash
pip install brucezilany
```

---

## 🧪 Example Usage

```python
import numpy as np
from brucezilany import inner_hair_cell, synapse, stimulus, map_to_synapse

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
ihc_output = inner_hair_cell(stim, cf=1000, n_rep=10)

# Intermediate mapper. In the orginal Matlab, this was part of the synapse function. 
# Here, it is separate, and needs to be called before synapse.
mapped_ihc = map_to_synapse(
    ihc_output=ihc_output,
    spontaneous_firing_rate=100,
    characteristic_frequency=1000,
    time_resolution=stim.time_resolution,
)

# Synapse response
out = synapse(
    amplitude_ihc=mapped_ihc,
    cf=1000,
    n_rep=10,
    n_timesteps=stim.n_simulation_timesteps,
    spontaneous_firing_rate=100
)

print("Mean firing rate:", np.mean(out.mean_firing_rate))
```

See `examples/` for more. These contain Python translations of the Matlab test files contained in the original source.

---

## ⚙️ Model Components

### `inner_hair_cell`

Simulates the receptor potential of an inner hair cell (IHC) in response to acoustic stimuli. Includes cochlear filtering and nonlinear transduction.

**Args**:

* `stimulus`: `Stimulus` object
* `cf`: characteristic frequency (Hz)
* `n_rep`: number of stimulus repetitions
* `cohc`: normal ohc function (float)
* `cihc`: normal ihc function (float)
* `species`: `Species` enum


Returns:

* IHC output as a vector (float)

---

### `map_to_synapse`
In the original Matlab source, this was included in the `synapse` function. However, since you sometimes want to call `synapse` for a number of trails, see for example `test_adaptive_redocking.py`, you don't always need to recompute this function. Therefore, in this code, it is separate, and you call this outside of calling `synapse`. The function applies a mapping function to the output of `inner_hair_cell`, by default a `SOFTPLUS` function.

**Args**:

* `ihc_output`: array of floats
* `spontaneous_firing_rate`: spontaneous firing rate of the fiber
* `characteristic_frequency`: characteristic frequency
* `time_resolution`: 1 / sampling rate
* `mapping_function`: `SynapseMapping` enum


Returns:

* mapped IHC output as a vector (float)

---

### `synapse`

Simulates stochastic spike generation at the IHC-ANF synapse, using a 4-site vesicle model with adaptive redocking and realistic refractory behavior. Note that you can provide an optional seeded RNG. This is important, because otherwise this model will produce a deterministic output. Alternatively, calling `brucezilany.set_seed(<int>)` *before* calling `synapse`, also make this function behave randomly. Again, see  `test_adaptive_redocking.py` for an example.

**Args**:

* `amplitude_ihc`: IHC potential vector
* `cf`: characteristic frequency (Hz)
* `n_rep`: number of stimulus repetitions
* `n_timesteps`: number of stimulus time steps
* `time_resolution`: 1 / sampling rate
* `spontaneous_firing_rate`: in spikes/s
* `noise`: random noise mode (`NoiseType`)
* `pla_impl`: power-law adaptation method (`PowerLaw`)
* `spontaneous_firing_rate`: spontaneous firing rate of the fiber
* `abs_refractory_period`: absolute refractory period
* `rel_refractory_period`: relative refractory period
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
    duration=0.05,              # duration of tone burst (s)
    simulation_duration=0.1,    # total simulation time (s)
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
from brucezilany import Neurogram

ng = Neurogram(n_cf=50, n_low=5, n_med=5, n_high=10)
ng.create(sound_wave=stim, species=Species.HUMAN_SHERA, n_trials=3)

output = ng.get_output()  # 3D array: [fiber, trial, time]
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

  * Split between IHC, map_to_synapse, synapse
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


