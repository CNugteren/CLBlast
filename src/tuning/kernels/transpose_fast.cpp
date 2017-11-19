
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the transpose OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TuneTranspose {
 public:

  // Settings for this kernel (default command-line arguments)
  static TunerDefaults GetTunerDefaults() {
    auto settings = TunerDefaults();
    settings.options = {kArgM, kArgN, kArgAlpha};
    settings.default_m = 1024;
    settings.default_n = 1024;
    return settings;
  }

  // Settings for this kernel (general)
  static TunerSettings GetTunerSettings(const Arguments<T> &args) {
    auto settings = TunerSettings();

    // Identification of the kernel
    settings.kernel_family = "transpose";
    settings.kernel_name = "TransposeMatrixFast";
    settings.sources =
#include "../src/kernels/level3/level3.opencl"
#include "../src/kernels/level3/transpose_fast.opencl"
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
    settings.mul_local = {{"TRA_DIM", "TRA_DIM"}};
    settings.div_global = {{"TRA_WPT", "TRA_WPT"}};

    // Sets the tuning parameters and their possible values
    settings.parameters = {
      {"TRA_DIM", {4, 8, 16, 32, 64}},
      {"TRA_WPT", {1, 2, 4, 8, 16}},
      {"TRA_PAD", {0, 1}},
      {"TRA_SHUFFLE", {0, 1}},
    };

    // Describes how to compute the performance metrics
    settings.metric_amount = 2 * args.m * args.n * GetBytes(args.precision);
    settings.performance_unit = "GB/s";

    return settings;
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }
  static std::vector<Constraint> SetConstraints() { return {}; }

  // Sets the kernel's arguments
  static void SetArguments(Kernel &kernel, const Arguments<T> &args,
                           std::vector<Buffer<T>>& buffers) {
    kernel.SetArgument(0, static_cast<int>(args.m));
    kernel.SetArgument(1, buffers[2]()); // 2 == A matrix
    kernel.SetArgument(2, buffers[3]()); // 3 == B matrix
    kernel.SetArgument(3, GetRealArg(args.alpha));
  }
};

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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneTranspose<half>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneTranspose<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneTranspose<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneTranspose<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneTranspose<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
