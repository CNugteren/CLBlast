
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xger OpenCL kernels.
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
class TuneXger {
 public:

  // The representative kernel and the source code
  static std::string KernelFamily() { return "xger"; }
  static std::string KernelName() { return "Xger"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level2/level2.opencl"
      #include "../src/kernels/level2/xger.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() { return {kArgN, kArgM, kArgAlpha}; }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 1024; }
  static size_t DefaultN() { return 1024; }
  static size_t DefaultK() { return 1; } // N/A for this kernel
  static size_t DefaultBatchCount() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return 1.0; } // N/A for this kernel
  static size_t DefaultNumRuns() { return 10; } // run every kernel this many times for averaging
  static size_t DefaultNumSearchStragegy() { return 1; } // N/A for this kernel
  static size_t DefaultSwarmSizePSO() { return 8; } // N/A for this kernel
  static double DefaultInfluenceGlobalPSO(){ return 0.1; }// N/A for this kernel
  static double DefaultInfluenceLocalPSO(){ return 0.3; } // N/A for this kernel
  static double DefaultInfluenceRandomPSO(){ return 0.6; }// N/A for this kernel
  static size_t DefaultHeuristic(){ return size_t{0};}// Full search
  static double DefaultMaxTempAnn(){ return 1.0;}// N/A for this kernel
  
  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) { return args.m; }
  static size_t GetSizeY(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeA(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeB(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeC(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    tuner.AddParameter(id, "WGS1", {4, 8, 16, 32, 64, 128, 256, 512});
    tuner.AddParameter(id, "WGS2", {1, 2, 4, 8, 16, 32, 64, 128, 256});
    tuner.AddParameter(id, "WPT", {1, 2, 4});
  }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &, const size_t, const Arguments<T> &) { }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &args) { return {args.m, args.n}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &args) { return GlobalSize(args); }
  static std::vector<size_t> LocalSize() { return {1, 1}; }
  static std::vector<size_t> LocalSizeRef() { return {8, 8}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"WGS1", "WGS2"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {}; }
  static TransformVector DivGlobal() { return {{"WPT", "WPT"}}; }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &x_vec, std::vector<T> &y_vec,
                           std::vector<T> &a_mat, std::vector<T> &, std::vector<T> &,
                           std::vector<T> &) {
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentInput(x_vec);
    tuner.AddArgumentScalar(0); // x_offset
    tuner.AddArgumentScalar(1); // x_increment
    tuner.AddArgumentInput(y_vec);
    tuner.AddArgumentScalar(0); // y_offset
    tuner.AddArgumentScalar(1); // y_increment
    tuner.AddArgumentOutput(a_mat);
    tuner.AddArgumentScalar(0); // a_offset
    tuner.AddArgumentScalar(static_cast<int>(args.m)); // a_ld
    tuner.AddArgumentScalar(0); // a_is_rowmajor
  }

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return (2*args.m*args.n + args.m + args.n) * GetBytes(args.precision);
  }
  static std::string PerformanceUnit() { return "GB/s"; }

  // Returns which Heuristic to run 
  static size_t GetCurrentHeuristic(const Arguments<T> &args){
    return size_t{0}; // Full search
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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXger<half>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXger<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXger<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXger<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXger<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
