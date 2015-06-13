
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements an auto-tuner to tune the Xgemv OpenCL kernel. It uses the CLTune library.
//
// =================================================================================================

#include <string>
#include <vector>
#include <stdexcept>

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// The Xgemv auto-tuner
template <typename T>
void XgemvTune(const Arguments<T> &args,
               const std::vector<T> &a_mat, const std::vector<T> &x_vec, std::vector<T> &y_vec,
               cltune::Tuner &tuner) {

  // This points to the Xgemv kernel as found in the CLBlast library
  std::string common_source =
  #include "../src/kernels/common.opencl"
  std::string kernel_source =
  #include "../src/kernels/xgemv.opencl"
  auto sources = common_source + kernel_source;
  auto id = tuner.AddKernelFromString(sources, "XgemvFast", {args.m}, {1});
  tuner.SetReferenceFromString(sources, "Xgemv", {args.m}, {64});

  // Sets the tunable parameters and their possible values
  tuner.AddParameter(id, "WGS", {64, 128, 256, 512, 1024, 1536, 2048});
  tuner.AddParameter(id, "WPT", {1, 2, 4, 8});
  tuner.AddParameter(id, "VW", {1, 2, 4, 8});

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Sets the constraints
  auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
  tuner.AddConstraint(id, MultipleOfX, {"WGS", "VW"});
  tuner.AddConstraint(id, MultipleOfX, {"WPT", "VW"});

  // Modifies the thread-sizes (local) based on the parameters
  tuner.MulLocalSize(id, {"WGS"});
  tuner.DivGlobalSize(id, {"WPT"});

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(args.alpha);
  tuner.AddArgumentScalar(args.beta);
  tuner.AddArgumentScalar(static_cast<int>(args.layout));
  tuner.AddArgumentInput(a_mat);
  tuner.AddArgumentScalar(0);
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentInput(x_vec);
  tuner.AddArgumentScalar(0);
  tuner.AddArgumentScalar(1);
  tuner.AddArgumentOutput(y_vec);
  tuner.AddArgumentScalar(0);
  tuner.AddArgumentScalar(1);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void TunerXgemv(int argc, char *argv[]) {
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: TunerAXY<float>(argc, argv, XgemvTune<float>); break;
    case Precision::kDouble: TunerAXY<double>(argc, argv, XgemvTune<double>); break;
    case Precision::kComplexSingle: TunerAXY<float2>(argc, argv, XgemvTune<float2>); break;
    case Precision::kComplexDouble: TunerAXY<double2>(argc, argv, XgemvTune<double2>); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::TunerXgemv(argc, argv);
  return 0;
}

// =================================================================================================
