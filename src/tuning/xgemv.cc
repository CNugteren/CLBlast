
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements an auto-tuner to tune the Xgemv OpenCL kernel. It uses the CLTune library.
// Three variations of the kernel are tuned:
// 1: The full version of the kernel
// 2: The fast version for non-transposed matrices
// 3: The fast version for transposed matrices
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
void XgemvTune(const Arguments<T> &args, const size_t variation,
               const std::vector<T> &a_mat, const std::vector<T> &x_vec, std::vector<T> &y_vec,
               cltune::Tuner &tuner) {

  // Sets the kernel name and the layout argument
  auto kernel_name = (variation == 1) ? "Xgemv" : ((variation == 2) ? "XgemvFast" : "XgemvFastRot");
  auto a_rotated = (variation == 3) ? 1 : 0;

  // This points to the Xgemv kernel as found in the CLBlast library
  std::string sources =
    #include "../src/kernels/common.opencl"
    #include "../src/kernels/xgemv.opencl"
  ;
  auto id = tuner.AddKernelFromString(sources, kernel_name, {args.m}, {1});
  tuner.SetReferenceFromString(sources, "Xgemv", {args.m}, {64});

  // Helper for the constraints
  auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };

  // Sets the tunable parameters, their possible values, the adjusted thread sizes, and constraints
  if (variation == 1) {
    tuner.AddParameter(id, "WGS1", {64, 128, 256, 512, 1024, 1536, 2048});
    tuner.AddParameter(id, "WPT1", {1, 2, 4, 8});
    tuner.MulLocalSize(id, {"WGS1"});
    tuner.DivGlobalSize(id, {"WPT1"});
  }
  else if (variation == 2) {
    tuner.AddParameter(id, "WGS2", {64, 128, 256, 512, 1024, 1536, 2048});
    tuner.AddParameter(id, "WPT2", {1, 2, 4, 8});
    tuner.AddParameter(id, "VW2", {1, 2, 4, 8});
    tuner.MulLocalSize(id, {"WGS2"});
    tuner.DivGlobalSize(id, {"WPT2"});
    tuner.AddConstraint(id, MultipleOfX, {"WPT2", "VW2"});
  }
  else if (variation == 3) {
    tuner.AddParameter(id, "WGS3", {64, 128, 256, 512, 1024, 1536, 2048});
    tuner.AddParameter(id, "WPT3", {1, 2, 4, 8});
    tuner.AddParameter(id, "VW3", {1, 2, 4, 8});
    tuner.MulLocalSize(id, {"WGS3"});
    tuner.DivGlobalSize(id, {"WPT3"});
    tuner.AddConstraint(id, MultipleOfX, {"WGS3", "VW3"});
  }

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(args.alpha);
  tuner.AddArgumentScalar(args.beta);
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
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void TunerXgemv(int argc, char *argv[]) {
  auto num_variations = size_t{3};
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: TunerAXY<float>(argc, argv, num_variations, XgemvTune<float>); break;
    case Precision::kDouble: TunerAXY<double>(argc, argv, num_variations, XgemvTune<double>); break;
    case Precision::kComplexSingle: TunerAXY<float2>(argc, argv, num_variations, XgemvTune<float2>); break;
    case Precision::kComplexDouble: TunerAXY<double2>(argc, argv, num_variations, XgemvTune<double2>); break;
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
