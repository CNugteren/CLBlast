
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the copy OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Settings for this kernel (default command-line arguments)
TunerDefaults CopyGetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgM, kArgN, kArgAlpha};
  settings.default_m = 1024;
  settings.default_n = 1024;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings CopyGetTunerSettings(const int, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "copy";
  settings.kernel_name = "CopyMatrixFast";
  settings.sources =
#include "../src/kernels/level3/level3.opencl"
#include "../src/kernels/level3/copy_fast.opencl"
  ;

  // Buffer sizes
  settings.size_a = args.m * args.n;
  settings.size_b = args.m * args.n;

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {2, 3};
  settings.outputs = {3};

  // Sets the base thread configuration
  settings.global_size = {args.m, args.n};
  settings.global_size_ref = settings.global_size;
  settings.local_size = {1, 1};
  settings.local_size_ref = {8, 8};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"COPY_DIMX", "COPY_DIMY"}};
  settings.div_global = {{"COPY_VW", "COPY_WPT"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
    {"COPY_DIMX", {8, 16, 32}},
    {"COPY_DIMY", {8, 16, 32}},
    {"COPY_WPT", {1, 2, 4, 8}},
    {"COPY_VW", {1, 2, 4, 8}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = 2 * args.m * args.n * GetBytes(args.precision);
  settings.performance_unit = "GB/s";

  return settings;
}

// Tests for valid arguments
template <typename T>
void CopyTestValidArguments(const int, const Arguments<T> &) { }
std::vector<Constraint> CopySetConstraints(const int) { return {}; }
template <typename T>
LocalMemSizeInfo CopyComputeLocalMemSize(const int) {
  return { [] (std::vector<size_t>) -> size_t { return 0; }, {} };
}

// Sets the kernel's arguments
template <typename T>
void CopySetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  kernel.SetArgument(0, static_cast<int>(args.m));
  kernel.SetArgument(1, buffers[2]()); // 2 == A matrix
  kernel.SetArgument(2, buffers[3]()); // 3 == B matrix
  kernel.SetArgument(3, GetRealArg(args.alpha));
}

// =================================================================================================
} // namespace clblast
