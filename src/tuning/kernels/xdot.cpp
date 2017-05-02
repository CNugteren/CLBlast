
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

  // The representative kernel and the source code
  static std::string KernelFamily() { return "xdot_"+std::to_string(V); }
  static std::string KernelName() { return (V==1) ? "Xdot" : "XdotEpilogue"; }
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
  static size_t DefaultN() { return 2*1024*1024; }
  static size_t DefaultK() { return 1; } // N/A for this kernel
  static size_t DefaultBatchCount() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return 1.0; } // N/A for this kernel
  static size_t DefaultNumRuns() { return 10; } // run every kernel this many times for averaging

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeY(const Arguments<T> &args) { return args.n; }
  static size_t GetSizeA(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeB(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeC(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeTemp(const Arguments<T> &args) { return args.n; } // Worst case

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    tuner.AddParameter(id, "WGS"+std::to_string(V), {32, 64, 128, 256, 512, 1024});
  }

  // Sets the constraints and local memory size
  static void SetConstraints(cltune::Tuner &, const size_t) { }
  static void SetLocalMemorySize(cltune::Tuner &, const size_t, const Arguments<T> &) { }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &) { return (V==1) ? std::vector<size_t>{2*64} : std::vector<size_t>{1}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &) { return (V==1) ? std::vector<size_t>{2*64*64} : std::vector<size_t>{64}; }
  static std::vector<size_t> LocalSize() { return {1}; }
  static std::vector<size_t> LocalSizeRef() { return {64}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return (V==1) ? TransformVector{{"WGS1"}} : TransformVector{{"WGS2"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return (V==1) ? TransformVector{{"WGS1"}} : TransformVector{{"WGS2"}}; }
  static TransformVector DivGlobal() { return {}; }

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

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return (V==1) ? (2*args.n + 1) * GetBytes(args.precision) : 1 * GetBytes(args.precision);
  }
  static std::string PerformanceUnit() { return (V==1) ? "GB/s" : "N/A"; }
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
