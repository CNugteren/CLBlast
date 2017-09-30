
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the padtranspose OpenCL kernels.
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
class TunePadTranspose {
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
    settings.kernel_family = "padtranspose";
    settings.kernel_name = "TransposePadMatrix";
    settings.sources =
#include "../src/kernels/common.opencl"
#include "../src/kernels/level3/level3.opencl"
#include "../src/kernels/level3/transpose_pad.opencl"
    ;

    // Buffer sizes
    settings.size_a = args.m * args.n;
    settings.size_b = args.m * args.n;

    // Sets the base thread configuration
    settings.global_size = {args.m, args.n};
    settings.global_size_ref = settings.global_size;
    settings.local_size = {1, 1};
    settings.local_size_ref = {8, 8};

    // Transforms the thread configuration based on the parameters
    settings.mul_local = {{"PADTRA_TILE", "PADTRA_TILE"}};
    settings.div_global = {{"PADTRA_WPT", "PADTRA_WPT"}};

    // Sets the tuning parameters and their possible values
    settings.parameters = {
      {"PADTRA_TILE", {8, 16, 32, 64}},
      {"PADTRA_WPT", {1, 2, 4, 8, 16}},
      {"PADTRA_PAD", {0, 1}},
    };

    // Describes how to compute the performance metrics
    settings.metric_amount = 2 * args.m * args.n * GetBytes(args.precision);
    settings.performance_unit = "GB/s";

    return settings;
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &tuner, const size_t id, const Arguments<T> &args) {
    auto LocalMemorySize = [args] (std::vector<size_t> v) {
      return ((v[0]*v[1]*(v[0]*v[1]+v[2]))*GetBytes(args.precision));
    };
    tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"PADTRA_TILE", "PADTRA_WPT", "PADTRA_PAD"});
  }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &, std::vector<T> &,
                           std::vector<T> &a_mat, std::vector<T> &b_mat, std::vector<T> &,
                           std::vector<T> &) {
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(0);
    tuner.AddArgumentInput(a_mat);
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(0);
    tuner.AddArgumentOutput(b_mat);
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentScalar(0);
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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TunePadTranspose<half>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TunePadTranspose<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TunePadTranspose<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TunePadTranspose<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TunePadTranspose<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
