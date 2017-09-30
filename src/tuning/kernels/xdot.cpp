
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xdot OpenCL kernels. Note that the results are
// not verified, since the result is not final and depends on the WGS2 parameter.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, int V>
class TuneXdot {
 public:

  // Settings for this kernel (default command-line arguments)
  static TunerDefaults GetTunerDefaults() {
    auto settings = TunerDefaults();
    settings.options = {kArgN};
    settings.default_n = 2*1024*1024;
    return settings;
  }

  // Settings for this kernel (general)
  static TunerSettings GetTunerSettings(const Arguments<T> &args) {
    auto settings = TunerSettings();

    // Identification of the kernel
    settings.kernel_family = "xdot_"+std::to_string(V);
    settings.kernel_name = (V==1) ? "Xdot" : "XdotEpilogue";
    settings.sources =
#include "../src/kernels/common.opencl"
#include "../src/kernels/level1/xdot.opencl"
    ;

    // Buffer sizes
    settings.size_x = args.n;
    settings.size_y = args.n;
    settings.size_temp = args.n; // Worst case

    // Sets the base thread configuration
    settings.global_size = (V==1) ? std::vector<size_t>{2*64} : std::vector<size_t>{1};
    settings.global_size_ref = (V==1) ? std::vector<size_t>{2*64*64} : std::vector<size_t>{64};
    settings.local_size = {1};
    settings.local_size_ref = {64};

    // Transforms the thread configuration based on the parameters
    settings.mul_local = (V==1) ? TunerSettings::TransformVector{{"WGS1"}} : TunerSettings::TransformVector{{"WGS2"}};
    settings.mul_global = (V==1) ? TunerSettings::TransformVector{{"WGS1"}} : TunerSettings::TransformVector{{"WGS2"}};

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
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &, const size_t, const Arguments<T> &) { }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &x_vec, std::vector<T> &y_vec,
                           std::vector<T> &, std::vector<T> &, std::vector<T> &,
                           std::vector<T> &temp) {
    if (V == 1) {
      tuner.AddArgumentScalar(static_cast<int>(args.n));
      tuner.AddArgumentInput(x_vec);
      tuner.AddArgumentScalar(0);
      tuner.AddArgumentScalar(1);
      tuner.AddArgumentInput(y_vec);
      tuner.AddArgumentScalar(0);
      tuner.AddArgumentScalar(1);
      tuner.AddArgumentInput(temp); // No output checking for the result - size varies
      tuner.AddArgumentScalar(static_cast<int>(false));
    }
    else {
      tuner.AddArgumentInput(temp);
      tuner.AddArgumentInput(x_vec); // No output checking for the result - store somewhere
      tuner.AddArgumentScalar(0);
    }
  }
};

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXdot<half, V>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXdot<float, V>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXdot<double, V>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXdot<float2, V>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXdot<double2, V>, double2>(argc, argv); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  return 0;
}

// =================================================================================================
