# Bruce-Zilany-Carney Auditory Nerve Model (Python Interface)

This repository provides a modern Python interface to the auditory periphery model developed by the Bruce, Zilany, and Carney labs. The model simulates spike train responses in auditory nerve fibers with detailed physiological realism.

It is implemented in **C++ with `pybind11`**, offering performance and modularity. All dependencies on Matlab have been removed, making it fully standalone and suitable for Python-based environments (e.g., ML, neurophysiology toolkits).

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

## ✅ Key Features

- **Standalone** C++ codebase — no Matlab needed
- **Modernized C++14** — no raw pointers, RAII memory management
- **Multi-threaded `Neurogram` class** — high-throughput auditory simulation across fibers
- Clean Python bindings using **`pybind11`**
- Full support for stimulus generation, IHC modeling, and synapse simulation

---

## 🚀 Getting Started

### 1. Install prerequisites

You need a working C++14 compiler and Python 3.7+. Use a virtual environment if needed.

### 2. Clone and install

```bash
git clone https://github.com/yourusername/zilany-an-model-python.git
cd zilany-an-model-python
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
from brucecpp import inner_hair_cell, synapse, stimulus

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


