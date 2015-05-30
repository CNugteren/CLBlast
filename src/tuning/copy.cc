
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements an auto-tuner to tune the copy OpenCL kernels. It uses CLTune.
//
// =================================================================================================

#include <string>
#include <vector>
#include <stdexcept>

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// The copy auto-tuner
template <typename T>
void CopyTune(const Arguments<T> &args,
              const std::vector<T> &a_mat, std::vector<T> &b_mat,
              cltune::Tuner &tuner) {

  // This points to the CopyMatrix kernel as found in the CLBlast library. This is just one example
  // of a copy kernel. However, all copy-kernels use the same tuning parameters, so one has to be
  // chosen as a representative.
  std::string common_source =
  #include "../src/kernels/common.opencl"
  std::string kernel_source =
  #include "../src/kernels/copy.opencl"
  auto sources = common_source + kernel_source;
  auto id = tuner.AddKernelFromString(sources, "CopyMatrix", {args.m, args.n}, {1, 1});
  tuner.SetReferenceFromString(sources, "CopyMatrix", {args.m, args.n}, {8, 8});

  // Sets the tunable parameters and their possible values
  tuner.AddParameter(id, "COPY_DIMX", {8, 16, 32});
  tuner.AddParameter(id, "COPY_DIMY", {8, 16, 32});
  tuner.AddParameter(id, "COPY_WPT", {1, 2, 4, 8});
  tuner.AddParameter(id, "COPY_VW", {1, 2, 4, 8});

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Modifies the thread-sizes (both global and local) based on the parameters
  tuner.MulLocalSize(id, {"COPY_DIMX", "COPY_DIMY"});
  tuner.DivGlobalSize(id, {"COPY_VW", "COPY_WPT"});

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentInput(a_mat);
  tuner.AddArgumentOutput(b_mat);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void TunerCopy(int argc, char *argv[]) {
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: TunerAB<float>(argc, argv, CopyTune<float>); break;
    case Precision::kDouble: TunerAB<double>(argc, argv, CopyTune<double>); break;
    case Precision::kComplexSingle: TunerAB<float2>(argc, argv, CopyTune<float2>); break;
    case Precision::kComplexDouble: TunerAB<double2>(argc, argv, CopyTune<double2>); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::TunerCopy(argc, argv);
  return 0;
}

// =================================================================================================
