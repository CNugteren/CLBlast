
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xger OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Settings for this kernel (default command-line arguments)
TunerDefaults XgerGetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgM, kArgN, kArgAlpha};
  settings.default_m = 1024;
  settings.default_n = 1024;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings XgerGetTunerSettings(const int, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "xger";
  settings.kernel_name = "Xger";
  settings.sources =
#include "../src/kernels/level2/level2.opencl"
#include "../src/kernels/level2/xger.opencl"
  ;

  // Buffer sizes
  settings.size_x = args.m;
  settings.size_y = args.n;
  settings.size_a = args.m * args.n;

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {0, 1, 2};
  settings.outputs = {2};

  // Sets the base thread configuration
  settings.global_size = {args.m, args.n};
  settings.global_size_ref = settings.global_size;
  settings.local_size = {1, 1};
  settings.local_size_ref = {8, 8};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"WGS1", "WGS2"}};
  settings.div_global = {{"WPT", "WPT"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
    {"WGS1", {4, 8, 16, 32, 64, 128, 256, 512}},
    {"WGS2", {1, 2, 4, 8, 16, 32, 64, 128, 256}},
    {"WPT", {1, 2, 4}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = (2*args.m*args.n + args.m + args.n) * GetBytes(args.precision);
  settings.performance_unit = "GB/s";

  return settings;
}

// Tests for valid arguments
template <typename T>
void XgerTestValidArguments(const int, const Arguments<T> &) { }
std::vector<Constraint> XgerSetConstraints(const int) { return {}; }
template <typename T>
LocalMemSizeInfo XgerComputeLocalMemSize(const int) {
  return { [] (std::vector<size_t>) -> size_t { return 0; }, {} };
}

// Sets the kernel's arguments
template <typename T>
void XgerSetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  kernel.SetArgument(0, static_cast<int>(args.m));
  kernel.SetArgument(1, static_cast<int>(args.n));
  kernel.SetArgument(2, GetRealArg(args.alpha));
  kernel.SetArgument(3, buffers[0]()); // 0 == X vector
  kernel.SetArgument(4, 0); // x_offset
  kernel.SetArgument(5, 1); // x_increment
  kernel.SetArgument(6, buffers[1]()); // 1 == Y vector
  kernel.SetArgument(7, 0); // y_offset
  kernel.SetArgument(8, 1); // y_increment
  kernel.SetArgument(9, buffers[2]()); // 2 == A matrix
  kernel.SetArgument(10, 0); // a_offset
  kernel.SetArgument(11, static_cast<int>(args.m)); // a_ld
  kernel.SetArgument(12, 0); // a_is_rowmajor
}

// =================================================================================================
} // namespace clblast
