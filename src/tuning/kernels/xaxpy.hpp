
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xaxpy OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Settings for this kernel (default command-line arguments)
TunerDefaults XaxpyGetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgN, kArgAlpha};
  settings.default_n = 4096*1024;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings XaxpyGetTunerSettings(const int, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "xaxpy";
  settings.kernel_name = "XaxpyFastest";
  settings.sources =
#include "../src/kernels/level1/level1.opencl"
#include "../src/kernels/level1/xaxpy.opencl"
  ;

  // Buffer sizes
  settings.size_x = args.n;
  settings.size_y = args.n;

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {0, 1};
  settings.outputs = {1};

  // Sets the base thread configuration
  settings.global_size = {args.n};
  settings.global_size_ref = settings.global_size;
  settings.local_size = {1};
  settings.local_size_ref = {64};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"WGS"}};
  settings.div_global = {{"WPT"},{"VW"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
    {"WGS", {64, 128, 256, 512, 1024, 2048}},
    {"WPT", {1, 2, 4, 8}},
    {"VW", {1, 2, 4, 8}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = 3 * args.n * GetBytes(args.precision);
  settings.performance_unit = "GB/s";

  return settings;
}

// Tests for valid arguments
template <typename T>
void XaxpyTestValidArguments(const int, const Arguments<T> &args) {
  if (!IsMultiple(args.n, 64)) {
    throw std::runtime_error("'XaxpyFastest' requires 'n' to be a multiple of WGS*WPT*VW");
  }
}
std::vector<Constraint> XaxpySetConstraints(const int) { return {}; }
template <typename T>
LocalMemSizeInfo XaxpyComputeLocalMemSize(const int) {
  return { [] (std::vector<size_t>) -> size_t { return 0; }, {} };
}

// Sets the kernel's arguments
template <typename T>
void XaxpySetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  kernel.SetArgument(0, static_cast<int>(args.n));
  kernel.SetArgument(1, GetRealArg(args.alpha));
  kernel.SetArgument(2, buffers[0]()); // 0 == X vector
  kernel.SetArgument(3, buffers[1]()); // 1 == Y vector
}

// =================================================================================================
} // namespace clblast
