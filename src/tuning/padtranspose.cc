
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements an auto-tuner to tune the pad-transpose OpenCL kernels. It uses CLTune.
//
// =================================================================================================

#include <string>
#include <vector>
#include <stdexcept>

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// The transpose auto-tuner
template <typename T>
void PadTransposeTune(const Arguments<T> &args,
                      const std::vector<T> &a_mat, std::vector<T> &b_mat,
                      cltune::Tuner &tuner) {

  // This points to the PadTransposeMatrix kernel as found in the CLBlast library. This is just one
  // example of a transpose kernel. However, all kernels use the same tuning parameters, so one has
  // to be chosen as a representative.
  std::string common_source =
  #include "../src/kernels/common.opencl"
  std::string kernel_source =
  #include "../src/kernels/padtranspose.opencl"
  auto sources = common_source + kernel_source;
  auto id = tuner.AddKernelFromString(sources, "PadTransposeMatrix", {args.m, args.n}, {1, 1});
  tuner.SetReferenceFromString(sources, "PadTransposeMatrix", {args.m, args.n}, {8, 8});

  // Sets the tunable parameters and their possible values
  tuner.AddParameter(id, "PADTRA_TILE", {8, 16, 32, 64});
  tuner.AddParameter(id, "PADTRA_WPT", {1, 2, 4, 8, 16});
  tuner.AddParameter(id, "PADTRA_PAD", {0, 1});

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Sets the constraints for local memory size limitations
  auto LocalMemorySize = [args] (std::vector<size_t> v) {
    return ((v[0]*v[1]*(v[0]*v[1]+v[2]))*GetBytes(args.precision));
  };
  tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"PADTRA_TILE", "PADTRA_WPT", "PADTRA_PAD"});

  // Modifies the thread-sizes (both global and local) based on the parameters
  tuner.DivGlobalSize(id, {"PADTRA_WPT", "PADTRA_WPT"});
  tuner.MulLocalSize(id, {"PADTRA_TILE", "PADTRA_TILE"});

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentScalar(0);
  tuner.AddArgumentInput(a_mat);
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(0);
  tuner.AddArgumentOutput(b_mat);
  tuner.AddArgumentScalar(0);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void TunerPadTranspose(int argc, char *argv[]) {
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: TunerAB<float>(argc, argv, PadTransposeTune<float>); break;
    case Precision::kDouble: TunerAB<double>(argc, argv, PadTransposeTune<double>); break;
    case Precision::kComplexSingle: TunerAB<float2>(argc, argv, PadTransposeTune<float2>); break;
    case Precision::kComplexDouble: TunerAB<double2>(argc, argv, PadTransposeTune<double2>); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::TunerPadTranspose(argc, argv);
  return 0;
}

// =================================================================================================
