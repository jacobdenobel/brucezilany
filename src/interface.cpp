#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bruce.h"

namespace py = pybind11;

void define_types(py::module &m)
{
    py::enum_<Species>(m, "Species", py::arithmetic())
        .value("CAT", CAT)
        .value("HUMAN_SHERA", HUMAN_SHERA)
        .value("HUMAN_GLASSBERG_MOORE", HUMAN_GLASSBERG_MOORE)
        .export_values();

    py::enum_<SynapseMapping>(m, "SynapseMapping", py::arithmetic())
        .value("NONE", NONE)
        .value("SOFTPLUS", SOFTPLUS)
        .value("EXPONENTIAL", EXPONENTIAL)
        .value("BOLTZMAN", BOLTZMAN)
        .export_values();

    py::enum_<NoiseType>(m, "NoiseType", py::arithmetic())
        .value("ONES", ONES)
        .value("FIXED_MATLAB", FIXED_MATLAB)
        .value("FIXED_SEED", FIXED_SEED)
        .value("RANDOM", RANDOM)
        .export_values();

    py::enum_<PowerLaw>(m, "PowerLaw", py::arithmetic())
        .value("APPROXIMATED", APPROXIMATED)
        .value("ACTUAL", ACTUAL)
        .export_values();
}

void define_stimulus(py::module m)
{
    using namespace stimulus;
    py::class_<Stimulus>(m, "Stimulus")
        .def(py::init<const std::vector<double> &, size_t, double>(), py::arg("data"), py::arg("sampling_rate"), py::arg("simulation_duration"))
        .def_property_readonly("data", [](const Stimulus& self) {
            return py::array(self.data.size(), self.data.data());
        })
        .def_readonly("sampling_rate", &Stimulus::sampling_rate)
        .def_readonly("time_resolution", &Stimulus::time_resolution)
        .def_readonly("stimulus_duration", &Stimulus::stimulus_duration)
        .def_readonly("simulation_duration", &Stimulus::simulation_duration)
        .def_readonly("n_stimulation_timesteps", &Stimulus::n_stimulation_timesteps)
        .def_readonly("n_simulation_timesteps", &Stimulus::n_simulation_timesteps)
        .def("__repr__", [](const Stimulus &self)
             { return "<Stimulus (" + std::to_string(self.stimulus_duration) + "s " + std::to_string(self.sampling_rate) + " Hz)>"; });

    m.def("from_file", &from_file, py::arg("path"), py::arg("verbose") = false, py::arg("sim_time") = 1.0, py::arg("normalize") = true);
    m.def("ramped_sine_wave", &ramped_sine_wave,
          py::arg("duration"),
          py::arg("simulation_duration"),
          py::arg("sampling_rate"),
          py::arg("rt"),
          py::arg("delay"),
          py::arg("f0"),
          py::arg("db"));
    m.def("normalize_db", &normalize_db, py::arg("stim"), py::arg("stim_db") = 65);
}

py::array_t<double> create_2d_numpy_array(const std::vector<std::vector<double>> &vec)
{
    // Get the dimensions of the input vector
    size_t rows = vec.size();
    size_t cols = vec.empty() ? 0 : vec[0].size();

    // Allocate a new numpy array
    py::array_t<double> result({rows, cols});

    // Get a pointer to the data in the numpy array
    double *result_ptr = static_cast<double *>(result.request().ptr);

    // Copy the data from the vector to the numpy array
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result_ptr[i * cols + j] = vec[i][j];
        }
    }

    return result;
}

py::array_t<double> vector_to_numpy(const std::vector<std::vector<std::vector<double>>>& vec) {
    // Get the shape of the vector (Depth, Rows, Columns)
    ssize_t depth = vec.size();
    ssize_t rows = vec[0].size();
    ssize_t cols = vec[0][0].size();
    
    std::array<ssize_t, 3> shape = {depth, rows, cols};

    // Create NumPy array
    py::array_t<double> arr(shape);
    double* ptr = arr.mutable_data();
    
    // Fill the NumPy array with data from the vector
    for (size_t i = 0; i < depth; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                ptr[i * rows * cols + j * cols + k] = vec[i][j][k];
            }
        }
    }

    return arr;
}

void define_helper_objects(py::module m)
{
    py::class_<syn::SynapseOutput>(m, "SynapseOutput")
        .def(py::init<int, int>(), py::arg("n_rep"), py::arg("n_timesteps"))
        .def_readonly("n_rep", &syn::SynapseOutput::n_rep)
        .def_readonly("n_timesteps", &syn::SynapseOutput::n_timesteps)
        .def_readonly("n_total_timesteps", &syn::SynapseOutput::n_total_timesteps)
        .def_readonly("psth", &syn::SynapseOutput::psth)
        .def_readonly("synaptic_output", &syn::SynapseOutput::synaptic_output)
        .def_readonly("redocking_time", &syn::SynapseOutput::redocking_time)
        .def_readonly("spike_times", &syn::SynapseOutput::spike_times)
        .def_readonly("mean_firing_rate", &syn::SynapseOutput::mean_firing_rate)
        .def_readonly("variance_firing_rate", &syn::SynapseOutput::variance_firing_rate)
        .def_readonly("mean_relative_refractory_period", &syn::SynapseOutput::mean_relative_refractory_period);

    py::enum_<FiberType>(m, "FiberType", py::arithmetic())
        .value("LOW", LOW)
        .value("MEDIUM", MEDIUM)
        .value("HIGH", HIGH)
        .export_values();

    py::class_<Fiber>(m, "Fiber")
        .def(py::init<double, double, double, FiberType>(), py::arg("spont"), py::arg("tabs"), py::arg("trel"), py::arg("type"))
        .def_readwrite("spont", &Fiber::spont)
        .def_readwrite("tabs", &Fiber::tabs)
        .def_readwrite("trel", &Fiber::trel)
        .def_readwrite("type", &Fiber::type);

    py::class_<Neurogram>(m, "Neurogram")
        .def(py::init<size_t, size_t, size_t, size_t, int>(),
             py::arg("n_cf") = 40,
             py::arg("n_low") = 10,
             py::arg("n_med") = 10,
             py::arg("n_high") = 30,
             py::arg("n_threads") = -1)
        .def(py::init<std::vector<double>, size_t, size_t, size_t, int>(),
             py::arg("cfs"),
             py::arg("n_low") = 10,
             py::arg("n_med") = 10,
             py::arg("n_high") = 30,
             py::arg("n_threads") = -1
            )
        .def("create", &Neurogram::create,
             py::arg("sound_wave"),
             py::arg("n_rep") = 1,
             py::arg("n_trials") = 1,
             py::arg("species") = HUMAN_SHERA,
             py::arg("noise_type") = RANDOM,
             py::arg("power_law") = APPROXIMATED
        )
        .def("get_fibers", &Neurogram::get_fibers, py::arg("cf_idx"))
        .def("get_output", [](const Neurogram &self)
             { return vector_to_numpy(self.get_output()); })
        .def("get_cfs", [](const Neurogram &self){
            const auto x = self.get_cfs();
            return py::array(x.size(), x.data());
        })
        .def_readwrite("bin_width", &Neurogram::bin_width);
}

void define_model_functions(py::module m)
{
    m.def("inner_hair_cell", &inner_hair_cell,
          py::arg("stimulus"),
          py::arg("cf") = 1e3,
          py::arg("n_rep") = 1,
          py::arg("cohc") = 1,
          py::arg("cihc") = 1,
          py::arg("species") = HUMAN_SHERA);

    m.def("map_to_synapse", &synapse_mapping::map,
          py::arg("ihc_output"),
          py::arg("spontaneous_firing_rate"),
          py::arg("characteristic_frequency"),
          py::arg("time_resolution"),
          py::arg("mapping_function") = SOFTPLUS);

    m.def("synapse", &synapse,
          py::arg("amplitude_ihc"),
          py::arg("cf"),
          py::arg("n_rep"),
          py::arg("n_timesteps"),
          py::arg("time_resolution") = 1 / 100e3,
          py::arg("noise") = RANDOM,
          py::arg("pla_impl") = APPROXIMATED,
          py::arg("spontaneous_firing_rate") = 100,
          py::arg("abs_refractory_period") = 0.7,
          py::arg("rel_refractory_period") = 0.6,
          py::arg("calculate_stats") = true,
          py::arg("rng") = std::nullopt
    );
}

void define_utils(py::module m)
{
    using namespace utils;
    m.def("set_seed", &set_seed);

    py::class_<RandomGenerator>(m, "RandomGenerator")
        .def(py::init<size_t>(), py::arg("seed") = SEED)
        .def("rand1", &RandomGenerator::rand1)
        .def("randn1", &RandomGenerator::randn1)
        .def("fill_gaussian", &RandomGenerator::fill_gaussian, py::arg("x"))
        ;

}

PYBIND11_MODULE(brucecpp, m)
{
    m.doc() = "Python wrapper for Bruce hearing model";
    define_types(m);
    define_utils(m);
    define_stimulus(m.def_submodule("stimulus"));
    define_helper_objects(m);
    define_model_functions(m);
}