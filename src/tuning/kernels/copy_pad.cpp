
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the pad OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// Settings for this kernel (default command-line arguments)
TunerDefaults GetTunerDefaults(const int) {
  auto settings = TunerDefaults();
  settings.options = {kArgM, kArgN, kArgAlpha};
  settings.default_m = 1024;
  settings.default_n = 1024;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings GetTunerSettings(const int, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "pad";
  settings.kernel_name = "CopyPadMatrix";
  settings.sources =
#include "../src/kernels/level3/level3.opencl"
#include "../src/kernels/level3/copy_pad.opencl"
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
  settings.mul_local = {{"PAD_DIMX", "PAD_DIMY"}};
  settings.div_global = {{"PAD_WPTX", "PAD_WPTY"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
    {"PAD_DIMX", {8, 16, 32}},
    {"PAD_DIMY", {8, 16, 32}},
    {"PAD_WPTX", {1, 2, 4}},
    {"PAD_WPTY", {1, 2, 4}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = 2 * args.m * args.n * GetBytes(args.precision);
  settings.performance_unit = "GB/s";

  return settings;
}

// Tests for valid arguments
template <typename T>
void TestValidArguments(const int, const Arguments<T> &) { }
std::vector<Constraint> SetConstraints(const int) { return {}; }

// Sets the kernel's arguments
template <typename T>
void SetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  kernel.SetArgument(0, static_cast<int>(args.m));
  kernel.SetArgument(1, static_cast<int>(args.n));
  kernel.SetArgument(2, static_cast<int>(args.m));
  kernel.SetArgument(3, 0);
  kernel.SetArgument(4, buffers[2]()); // 2 == A matrix
  kernel.SetArgument(5, static_cast<int>(args.m));
  kernel.SetArgument(6, static_cast<int>(args.n));
  kernel.SetArgument(7, static_cast<int>(args.m));
  kernel.SetArgument(8, 0);
  kernel.SetArgument(9, buffers[3]()); // 3 == B matrix
  kernel.SetArgument(10, GetRealArg(args.alpha));
  kernel.SetArgument(11, 0);
}

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<half>, clblast::TestValidArguments<half>, clblast::SetConstraints, clblast::SetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<float>, clblast::TestValidArguments<float>, clblast::SetConstraints, clblast::SetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<double>, clblast::TestValidArguments<double>, clblast::SetConstraints, clblast::SetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<float2>, clblast::TestValidArguments<float2>, clblast::SetConstraints, clblast::SetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<double2>, clblast::TestValidArguments<double2>, clblast::SetConstraints, clblast::SetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
