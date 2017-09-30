
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xaxpy OpenCL kernels.
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
class TuneXaxpy {
 public:

  // Settings for this kernel (default command-line arguments)
  static TunerDefaults GetTunerDefaults() {
    auto settings = TunerDefaults();
    settings.options = {kArgN, kArgAlpha};
    settings.default_n = 4096*1024;
    return settings;
  }

  // Settings for this kernel (general)
  static TunerSettings GetTunerSettings(const Arguments<T> &args) {
    auto settings = TunerSettings();

    // Identification of the kernel
    settings.kernel_family = "xaxpy";
    settings.kernel_name = "XaxpyFastest";
    settings.sources =
#include "../src/kernels/common.opencl"
#include "../src/kernels/level1/level1.opencl"
#include "../src/kernels/level1/xaxpy.opencl"
    ;

    // Buffer sizes
    settings.size_x = args.n;
    settings.size_y = args.n;

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
  static void TestValidArguments(const Arguments<T> &args) {
    if (!IsMultiple(args.n, 64)) {
      throw std::runtime_error("'XaxpyFastest' requires 'n' to be a multiple of WGS*WPT*VW");
    }
  }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &, const size_t, const Arguments<T> &) { }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &x_vec, std::vector<T> &y_vec,
                           std::vector<T> &, std::vector<T> &, std::vector<T> &,
                           std::vector<T> &) {
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentInput(x_vec);
    tuner.AddArgumentOutput(y_vec);
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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXaxpy<half>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXaxpy<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXaxpy<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXaxpy<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXaxpy<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
