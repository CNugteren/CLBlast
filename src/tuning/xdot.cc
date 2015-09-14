
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

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TuneXdot {
 public:

  // The representative kernel and the source code
  static std::string KernelFamily() { return "xdot"; }
  static std::string KernelName() { return "Xdot"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level1/xdot.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() { return {kArgN}; }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 1; } // N/A for this kernel
  static size_t DefaultN() { return 4096*1024; }
  static size_t DefaultK() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return 1.0; } // N/A for this kernel

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeY(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeA(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeB(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeC(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeTemp(const Arguments<T> &args) { return args.n; } // Worst case

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    tuner.AddParameter(id, "WGS1", {32, 64, 128, 256, 512, 1024});
    tuner.AddParameter(id, "WGS2", {32, 64, 128, 256, 512, 1024});
    tuner.AddParameter(id, "VW", {1});
  }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &, const size_t, const Arguments<T> &) { }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &) { return {2}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &) { return {2*64*64}; }
  static std::vector<size_t> LocalSize() { return {1}; }
  static std::vector<size_t> LocalSizeRef() { return {64}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"WGS1"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {{"WGS1"},{"WGS2"}}; }
  static TransformVector DivGlobal() { return {}; }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &x_vec, std::vector<T> &y_vec,
                           std::vector<T> &, std::vector<T> &, std::vector<T> &,
                           std::vector<T> &temp) {
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

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return (2*args.n + 1) * GetBytes(args.precision);
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
  switch(clblast::GetPrecision(argc, argv)) {
    case clblast::Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXdot<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXdot<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXdot<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXdot<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
