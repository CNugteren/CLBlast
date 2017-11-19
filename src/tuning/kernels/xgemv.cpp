
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xgemv OpenCL kernels. Three variants are tuned:
// 1: The full version of the kernel
// 2: The fast version for non-transposed matrices
// 3: The fast version for transposed matrices
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
class TuneXgemv {
 public:

  // Settings for this kernel (default command-line arguments)
  static TunerDefaults GetTunerDefaults() {
    auto settings = TunerDefaults();
    settings.options = {kArgM, kArgN, kArgAlpha, kArgBeta};
    settings.default_m = 2048;
    settings.default_n = 2048;
    return settings;
  }

  // Settings for this kernel (general)
  static TunerSettings GetTunerSettings(const Arguments<T> &args) {
    auto settings = TunerSettings();

    // Identification of the kernel
    settings.kernel_family = (V==1) ? "xgemv" : ((V==2) ? "xgemv_fast" : "xgemv_fast_rot");
    settings.kernel_name = (V==1) ? "Xgemv" : ((V==2) ? "XgemvFast" : "XgemvFastRot");
    settings.sources =
#include "../src/kernels/level2/xgemv.opencl"
#include "../src/kernels/level2/xgemv_fast.opencl"
    ;

    // Buffer sizes
    settings.size_x = args.n;
    settings.size_y = args.m;
    settings.size_a = args.m * args.n;

    // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
    settings.inputs = {0, 1, 2};
    settings.outputs = {1};

    // Sets the base thread configuration
    settings.global_size = {args.m};
    settings.global_size_ref = settings.global_size;
    settings.local_size = {1};
    settings.local_size_ref = {64};

    // Transforms the thread configuration based on the parameters
    settings.mul_local = {{"WGS"+std::to_string(V)}};
    settings.div_global = (V==1 || V==2) ? TransformVector{{"WPT"+std::to_string(V)}} : TransformVector{};

    // Sets the tuning parameters and their possible values
    if (V==1) {
      settings.parameters = {
        {"WGS"+std::to_string(V), {32, 64, 128, 256}},
        {"WPT"+std::to_string(V), {1, 2, 4}},
      };
    }
    if (V==2) {
      settings.parameters = {
        {"WGS"+std::to_string(V), {16, 32, 64, 128, 256}},
        {"WPT"+std::to_string(V), {1, 2, 4}},
        {"VW"+std::to_string(V), {1, 2, 4, 8}},
      };
    }
    if (V==3) {
      settings.parameters = {
        {"WGS"+std::to_string(V), {16, 32, 64, 128}},
        {"WPT"+std::to_string(V), {1, 2, 4, 8, 16, 32}},
        {"VW"+std::to_string(V), {1, 2, 4, 8}},
      };
    }

    // Describes how to compute the performance metrics
    settings.metric_amount = (args.m*args.n + 2*args.m + args.n) * GetBytes(args.precision);
    settings.performance_unit = "GB/s";

    return settings;
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }
  static std::vector<Constraint> SetConstraints() {
    auto constraints = std::vector<Constraint>();
    if (V==2 || V==3) {
      auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
      constraints.push_back({MultipleOfX, {"WPT"+std::to_string(V), "VW"+std::to_string(V)}});
    }
    if (V==3) {
      auto LargerOrEqual = [] (std::vector<size_t> v) { return v[0] >= v[1]; };
      constraints.push_back({LargerOrEqual, {"WGS"+std::to_string(V), "WPT"+std::to_string(V)}});
    }
    return constraints;
  }

  // Sets the kernel's arguments
  static void SetArguments(Kernel &kernel, const Arguments<T> &args,
                           std::vector<Buffer<T>>& buffers) {
    auto a_rotated = (V==3) ? 1 : 0;
    kernel.SetArgument(0, static_cast<int>(args.m));
    kernel.SetArgument(1, static_cast<int>(args.n));
    kernel.SetArgument(2, GetRealArg(args.alpha));
    kernel.SetArgument(3, GetRealArg(args.beta));
    kernel.SetArgument(4, a_rotated);
    kernel.SetArgument(5, buffers[2]()); // 2 == A matrix
    kernel.SetArgument(6, 0);
    kernel.SetArgument(7, static_cast<int>(args.m));
    kernel.SetArgument(8, buffers[0]()); // 0 == X vector
    kernel.SetArgument(9, 0);
    kernel.SetArgument(10, 1);
    kernel.SetArgument(11, buffers[1]()); // 1 == Y vector
    kernel.SetArgument(12, 0);
    kernel.SetArgument(13, 1);
    kernel.SetArgument(14, 0); // Conjugate transpose
    kernel.SetArgument(15, 0); // Additional parameter
    kernel.SetArgument(16, 0); // Banded 'kl'
    kernel.SetArgument(17, 0); // Banded 'ku'
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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXgemv<half,V>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXgemv<float,V>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXgemv<double,V>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXgemv<float2,V>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXgemv<double2,V>, double2>(argc, argv); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  StartVariation<3>(argc, argv);
  return 0;
}

// =================================================================================================
