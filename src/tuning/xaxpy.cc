
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements an auto-tuner to tune the Xaxpy OpenCL kernel. It uses the CLTune library.
//
// =================================================================================================

#include <string>
#include <vector>
#include <stdexcept>

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// The Xaxpy auto-tuner
template <typename T>
void XaxpyTune(const Arguments<T> &args,
               const std::vector<T> &x_vec, std::vector<T> &y_vec,
               cltune::Tuner &tuner) {

  // The XaxpyFast kernel only works under certain conditions. Check here whether the condition is
  // true for the reference kernel
  if (!IsMultiple(args.n, 64)) {
    throw std::runtime_error("The 'XaxpyFast' kernel requires 'n' to be a multiple of WGS*WPT*VW");
  }

  // This points to the XaxpyFast kernel as found in the CLBlast library
  std::string common_source =
  #include "../src/kernels/common.opencl"
  std::string kernel_source =
  #include "../src/kernels/xaxpy.opencl"
  auto sources = common_source + kernel_source;
  auto id = tuner.AddKernelFromString(sources, "XaxpyFast", {args.n}, {1});
  tuner.SetReferenceFromString(sources, "XaxpyFast", {args.n}, {64});

  // Sets the tunable parameters and their possible values
  tuner.AddParameter(id, "WGS", {64, 128, 256, 512, 1024, 2048});
  tuner.AddParameter(id, "WPT", {1, 2, 4, 8});
  tuner.AddParameter(id, "VW", {1, 2, 4, 8});

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Modifies the thread-sizes (local) based on the parameters
  tuner.MulLocalSize(id, {"WGS"});
  tuner.DivGlobalSize(id, {"WPT"});
  tuner.DivGlobalSize(id, {"VW"});

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(args.alpha);
  tuner.AddArgumentInput(x_vec);
  tuner.AddArgumentOutput(y_vec);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void TunerXaxpy(int argc, char *argv[]) {
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: TunerXY<float>(argc, argv, XaxpyTune<float>); break;
    case Precision::kDouble: TunerXY<double>(argc, argv, XaxpyTune<double>); break;
    case Precision::kComplexSingle: TunerXY<float2>(argc, argv, XaxpyTune<float2>); break;
    case Precision::kComplexDouble: TunerXY<double2>(argc, argv, XaxpyTune<double2>); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::TunerXaxpy(argc, argv);
  return 0;
}

// =================================================================================================
