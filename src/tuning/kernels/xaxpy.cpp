
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

  // The representative kernel and the source code
  static std::string KernelFamily() { return "xaxpy"; }
  static std::string KernelName() { return "XaxpyFast"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level1/level1.opencl"
      #include "../src/kernels/level1/xaxpy.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() { return {kArgN, kArgAlpha}; }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &args) {
    if (!IsMultiple(args.n, 64)) {
      throw std::runtime_error("'XaxpyFast' requires 'n' to be a multiple of WGS*WPT*VW");
    }
  }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 1; } // N/A for this kernel
  static size_t DefaultN() { return 4096*1024; }
  static size_t DefaultK() { return 1; } // N/A for this kernel
  static size_t DefaultBatchCount() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return 1.0; } // N/A for this kernel
  static size_t DefaultNumRuns() { return 2; } // run every kernel this many times for averaging

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeY(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeA(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeB(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeC(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    tuner.AddParameter(id, "WGS", {64, 128, 256, 512, 1024, 2048});
    tuner.AddParameter(id, "WPT", {1, 2, 4, 8});
    tuner.AddParameter(id, "VW", {1, 2, 4, 8});
  }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &, const size_t, const Arguments<T> &) { }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &args) { return {args.n}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &args) { return GlobalSize(args); }
  static std::vector<size_t> LocalSize() { return {1}; }
  static std::vector<size_t> LocalSizeRef() { return {64}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"WGS"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {}; }
  static TransformVector DivGlobal() { return {{"WPT"},{"VW"}}; }

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

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return 3 * args.n * GetBytes(args.precision);
  }
  static std::string PerformanceUnit() { return "GB/s"; }
};

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
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
