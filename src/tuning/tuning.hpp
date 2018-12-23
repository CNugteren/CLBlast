
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the generic CLBlast auto-tuner (inspired by CLTune). This is only used for
//  the optional and stand-alone tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_TUNING_H_
#define CLBLAST_TUNING_TUNING_H_

#include <vector>
#include <string>
#include <random>
#include <utility>
#include <algorithm>
#include <chrono>
#include <functional>

#include "utilities/utilities.hpp"
#include "utilities/compile.hpp"
#include "utilities/timing.hpp"
#include "tuning/configurations.hpp"

namespace clblast {
// =================================================================================================

// Structures for the tuners with all the default settings
struct TunerDefaults {

  // The list of arguments relevant for this routine
  std::vector<std::string> options = {};

  // Default sizes
  size_t default_m = 1;
  size_t default_n = 1;
  size_t default_k = 1;
  size_t channels = 1;
  size_t height = 1;
  size_t width = 1;
  size_t kernel_h = 3;
  size_t kernel_w = 3;
  size_t num_kernels = 1;
  size_t batch_count = 1;

  // Other defaults
  size_t default_batch_count = 1;
  size_t default_num_runs = 10; // run every kernel this many times for averaging
  double default_fraction = 1.0;
};

// Structures for the tuners with the remaining settings
struct TunerSettings {

  // The representative kernel and the source code
  std::string kernel_family;
  std::string kernel_name;
  std::string sources;

  // Describes how to obtain the sizes of the buffers
  size_t size_x = 1;
  size_t size_y = 1;
  size_t size_a = 1;
  size_t size_b = 1;
  size_t size_c = 1;
  size_t size_temp = 1;

  // Inputs and outputs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  std::vector<size_t> inputs = {};
  std::vector<size_t> outputs = {};

  // Sets the base thread configuration
  std::vector<size_t> global_size = {};
  std::vector<size_t> global_size_ref = {};
  std::vector<size_t> local_size = {};
  std::vector<size_t> local_size_ref = {};

  // Transforms the thread configuration based on the parameters
  TransformVector mul_local = {};
  TransformVector div_local = {};
  TransformVector mul_global = {};
  TransformVector div_global = {};

  // Sets the tuning parameters and their possible values
  std::vector<Parameter> parameters;

  // Describes how to compute the performance metrics
  size_t metric_amount = 0;
  std::string performance_unit = "N/A";
};

// =================================================================================================

struct TuningResult { std::string name; double score; Configuration config; };

void PrintTimingsToFileAsJSON(const std::string &filename,
                              const Device& device, const Platform& platform,
                              const std::vector<std::pair<std::string,std::string>> &metadata,
                              const std::vector<TuningResult>& tuning_results);

void print_separator(const size_t parameters_size);

// =================================================================================================

using GetTunerDefaultsFunc = std::function<TunerDefaults(const int V)>;
template <typename T>
using GetTunerSettingsFunc = std::function<TunerSettings(const int V, const Arguments<T> &args)>;
template <typename T>
using TestValidArgumentsFunc = std::function<void(const int V, const Arguments<T> &args)>;
using SetConstraintsFunc = std::function<std::vector<Constraint>(const int V)>;
template <typename T>
using ComputeLocalMemSizeFunc = std::function<LocalMemSizeInfo(const int V)>;
template <typename T>
using SetArgumentsFunc = std::function<void(const int V, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers)>;

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for all types of kernel families. Note that this is a header-only function so
// that it is automatically compiled for the various kernels (given as the 'C' template argument).
template <typename T>
void Tuner(int argc, char* argv[], const int V,
           GetTunerDefaultsFunc GetTunerDefaults,
           GetTunerSettingsFunc<T> GetTunerSettings,
           TestValidArgumentsFunc<T> TestValidArguments,
           SetConstraintsFunc SetConstraints,
           ComputeLocalMemSizeFunc<T> ComputeLocalMemSize,
           SetArgumentsFunc<T> SetArguments);

// Function to run the tuners through the CLBlast API, no I/O
template <typename T>
StatusCode TunerAPI(Queue &queue, const Arguments<T> &args, const int V,
                    const GetTunerDefaultsFunc GetTunerDefaults,
                    const GetTunerSettingsFunc<T> GetTunerSettings,
                    const TestValidArgumentsFunc<T> TestValidArguments,
                    const SetConstraintsFunc SetConstraints,
                    const ComputeLocalMemSizeFunc<T> ComputeLocalMemSize,
                    const SetArgumentsFunc<T> SetArguments,
                    std::unordered_map<std::string,size_t> &parameters);

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_TUNING_H_
#endif
