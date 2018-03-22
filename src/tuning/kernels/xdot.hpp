
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xdot OpenCL kernels. Note that the results are
// not verified, since the result is not final and depends on the WGS2 parameter.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Settings for this kernel (default command-line arguments)
TunerDefaults XdotGetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgN};
  settings.default_n = 2*1024*1024;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings XdotGetTunerSettings(const int V, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "xdot_"+std::to_string(V);
  settings.kernel_name = (V==1) ? "Xdot" : "XdotEpilogue";
  settings.sources =
#include "../src/kernels/level1/xdot.opencl"
  ;

  // Buffer sizes
  settings.size_x = args.n;
  settings.size_y = args.n;
  settings.size_temp = args.n; // Worst case

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {0, 1, 5};
  settings.outputs = {}; // no output checking

  // Sets the base thread configuration
  settings.global_size = (V==1) ? std::vector<size_t>{2*64} : std::vector<size_t>{1};
  settings.global_size_ref = (V==1) ? std::vector<size_t>{2*64*64} : std::vector<size_t>{64};
  settings.local_size = {1};
  settings.local_size_ref = {64};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = (V==1) ? TransformVector{{"WGS1"}} : TransformVector{{"WGS2"}};
  settings.mul_global = (V==1) ? TransformVector{{"WGS1"}} : TransformVector{{"WGS2"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
    {"WGS"+std::to_string(V), {32, 64, 128, 256, 512, 1024}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = (V==1) ? (2*args.n + 1) * GetBytes(args.precision) : 1 * GetBytes(args.precision);
  settings.performance_unit = (V==1) ? "GB/s" : "N/A";

  return settings;
}

// Tests for valid arguments
template <typename T>
void XdotTestValidArguments(const int, const Arguments<T> &) { }
std::vector<Constraint> XdotSetConstraints(const int) { return {}; }
template <typename T>
LocalMemSizeInfo XdotComputeLocalMemSize(const int V) {
  return {
      [] (std::vector<size_t> v) -> size_t {
          return GetBytes(PrecisionValue<T>()) * v[0];
      },
      {"WGS"+std::to_string(V)}
  };}

// Sets the kernel's arguments
template <typename T>
void XdotSetArguments(const int V, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  if (V == 1) {
    kernel.SetArgument(0, static_cast<int>(args.n));
    kernel.SetArgument(1, buffers[0]()); // 0 == X vector
    kernel.SetArgument(2, 0);
    kernel.SetArgument(3, 1);
    kernel.SetArgument(4, buffers[1]()); // 1 == Y vector
    kernel.SetArgument(5, 0);
    kernel.SetArgument(6, 1);
    kernel.SetArgument(7, buffers[5]()); // 5 == temp; no output checking - size varies
    kernel.SetArgument(8, static_cast<int>(false));
  }
  else {
    kernel.SetArgument(0, buffers[5]()); // 5 == temp
    kernel.SetArgument(1, buffers[0]()); // 0 == X vector; no output checking - size varies
    kernel.SetArgument(2, 0);
  }
}

// =================================================================================================
} // namespace clblast
