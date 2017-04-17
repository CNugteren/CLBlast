
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xgemv OpenCL kernels. Three variants are tuned:
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

  // The representative kernel and the source code
  static std::string KernelFamily() { return (V==1) ? "xgemv" : ((V==2) ? "xgemv_fast" : "xgemv_fast_rot"); }
  static std::string KernelName() { return (V==1) ? "Xgemv" : ((V==2) ? "XgemvFast" : "XgemvFastRot"); }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level2/xgemv.opencl"
      #include "../src/kernels/level2/xgemv_fast.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() { return {kArgM, kArgN, kArgAlpha, kArgBeta}; }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 2048; }
  static size_t DefaultN() { return 2048; }
  static size_t DefaultK() { return 1; } // N/A for this kernel
  static size_t DefaultBatchCount() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return 1.0; } // N/A for this kernel
  static size_t DefaultNumRuns() { return 2; } // run every kernel this many times for averaging

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeY(const Arguments<T> &args) { return args.m; }
  static size_t GetSizeA(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeB(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeC(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    if (V==1) {
      tuner.AddParameter(id, "WGS"+std::to_string(V), {32, 64, 128, 256});
      tuner.AddParameter(id, "WPT"+std::to_string(V), {1, 2, 4});
    }
    if (V==2) {
      tuner.AddParameter(id, "WGS"+std::to_string(V), {16, 32, 64, 128, 256});
      tuner.AddParameter(id, "WPT"+std::to_string(V), {1, 2, 4});
      tuner.AddParameter(id, "VW"+std::to_string(V), {1, 2, 4, 8});
    }
    if (V==3) {
      tuner.AddParameter(id, "WGS"+std::to_string(V), {16, 32, 64, 128});
      tuner.AddParameter(id, "WPT"+std::to_string(V), {1, 2, 4, 8, 16, 32});
      tuner.AddParameter(id, "VW"+std::to_string(V), {1, 2, 4, 8});
    }
  }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &tuner, const size_t id) {
    if (V==2 || V==3) {
      auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
      tuner.AddConstraint(id, MultipleOfX, {"WPT"+std::to_string(V), "VW"+std::to_string(V)});
    }
    if (V==3) {
      auto LargerOrEqual = [] (std::vector<size_t> v) { return v[0] >= v[1]; };
      tuner.AddConstraint(id, LargerOrEqual, {"WGS"+std::to_string(V), "WPT"+std::to_string(V)});
    }
  }
  static void SetLocalMemorySize(cltune::Tuner &tuner, const size_t id, const Arguments<T> &args) {
    if (V==1 || V==2) {
      auto LocalMemorySize = [args] (std::vector<size_t> v) { return v[0]*GetBytes(args.precision); };
      tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"WGS"+std::to_string(V)});
    }
    else {
      auto LocalMemorySize = [args] (std::vector<size_t> v) { return (v[0]*v[1] + v[1])*GetBytes(args.precision); };
      tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"WGS"+std::to_string(V), "WPT"+std::to_string(V)});
    }
  }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &args) { return {args.m}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &args) { return GlobalSize(args); }
  static std::vector<size_t> LocalSize() { return {1}; }
  static std::vector<size_t> LocalSizeRef() { return {64}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"WGS"+std::to_string(V)}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {}; }
  static TransformVector DivGlobal() {
    if (V==1 || V==2) return {{"WPT"+std::to_string(V)}};
    return {};
  }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &x_vec, std::vector<T> &y_vec,
                           std::vector<T> &a_mat, std::vector<T> &, std::vector<T> &,
                           std::vector<T> &) {
    auto a_rotated = (V==3) ? 1 : 0;
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentScalar(GetRealArg(args.beta));
    tuner.AddArgumentScalar(static_cast<int>(a_rotated));
    tuner.AddArgumentInput(a_mat);
    tuner.AddArgumentScalar(0);
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentInput(x_vec);
    tuner.AddArgumentScalar(0);
    tuner.AddArgumentScalar(1);
    tuner.AddArgumentOutput(y_vec);
    tuner.AddArgumentScalar(0);
    tuner.AddArgumentScalar(1);
    tuner.AddArgumentScalar(0); // Conjugate transpose
    tuner.AddArgumentScalar(0); // Additional parameter
    tuner.AddArgumentScalar(0); // Banded 'kl'
    tuner.AddArgumentScalar(0); // Banded 'ku'
  }

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return (args.m*args.n + 2*args.m + args.n) * GetBytes(args.precision);
  }
  static std::string PerformanceUnit() { return "GB/s"; }
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
