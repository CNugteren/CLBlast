
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the pad OpenCL kernels.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TunePad {
 public:

  // The representative kernel and the source code
  static std::string KernelFamily() { return "pad"; }
  static std::string KernelName() { return "CopyPadMatrix"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level3/level3.opencl"
      #include "../src/kernels/level3/copy_pad.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() { return {kArgM, kArgN, kArgAlpha}; }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 1024; }
  static size_t DefaultN() { return 1024; }
  static size_t DefaultK() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return 1.0; } // N/A for this kernel

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeY(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeA(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeB(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeC(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    tuner.AddParameter(id, "PAD_DIMX", {8, 16, 32});
    tuner.AddParameter(id, "PAD_DIMY", {8, 16, 32});
    tuner.AddParameter(id, "PAD_WPTX", {1, 2, 4});
    tuner.AddParameter(id, "PAD_WPTY", {1, 2, 4});
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
  static TransformVector MulLocal() { return {{"PAD_DIMX", "PAD_DIMY"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {}; }
  static TransformVector DivGlobal() { return {{"PAD_WPTX", "PAD_WPTY"}}; }

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
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(0);
    tuner.AddArgumentOutput(b_mat);
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentScalar(0);
  }

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return 2 * args.m * args.n * GetBytes(args.precision);
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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TunePad<half>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TunePad<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TunePad<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TunePad<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TunePad<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
