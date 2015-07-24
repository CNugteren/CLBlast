
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements an auto-tuner to tune the Xgemm OpenCL kernel. It uses the CLTune library.
// Note that this tuner uses random-search: running it multiple times or with a larger fraction
// argument might be neccessary to obtain good results.
//
// =================================================================================================

#include <string>
#include <vector>
#include <stdexcept>

#include "internal/utilities.h"
#include "internal/tuning.h"

namespace clblast {
// =================================================================================================

// The Xgemm auto-tuner
template <typename T>
void XgemmTune(const Arguments<T> &args,
               const std::vector<T> &a_mat, const std::vector<T> &b_mat, std::vector<T> &c_mat,
               cltune::Tuner &tuner) {

  // This points to the Xgemm kernel as found in the CLBlast library and its golden reference
  std::string sources =
    #include "../src/kernels/common.opencl"
    #include "../src/kernels/xgemm.opencl"
  ;
  auto id = tuner.AddKernelFromString(sources, "Xgemm", {args.m, args.n}, {1, 1});
  tuner.SetReferenceFromString(sources, "Xgemm", {args.m, args.n}, {8, 8});

  // Sets the tunable parameters and their possible values
  tuner.AddParameter(id, "MWG", {16, 32, 64, 128});
  tuner.AddParameter(id, "NWG", {16, 32, 64, 128});
  tuner.AddParameter(id, "KWG", {16, 32});
  tuner.AddParameter(id, "MDIMC", {8, 16, 32});
  tuner.AddParameter(id, "NDIMC", {8, 16, 32});
  tuner.AddParameter(id, "MDIMA", {8, 16, 32});
  tuner.AddParameter(id, "NDIMB", {8, 16, 32});
  tuner.AddParameter(id, "KWI", {2, 8});
  tuner.AddParameter(id, "VWM", {1, 2, 4, 8});
  tuner.AddParameter(id, "VWN", {1, 2, 4, 8});
  tuner.AddParameter(id, "STRM", {0, 1});
  tuner.AddParameter(id, "STRN", {0, 1});
  tuner.AddParameter(id, "SA", {0, 1});
  tuner.AddParameter(id, "SB", {0, 1});

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Sets the helper functions to implement the constraints below
  auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
  auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
  auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };

  // Sets constraints: Requirement for unrolling the KWG loop
  tuner.AddConstraint(id, MultipleOfX, {"KWG", "KWI"});

  // Sets constraints: Required for integer MWI and NWI
  tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMC", "VWM"});
  tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMC", "VWN"});

  // Sets constraints: Required for integer MWIA and NWIB
  tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMA", "VWM"});
  tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMB", "VWN"});

  // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
  tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "MDIMA"});
  tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "NDIMB"});

  // Sets the constraints for local memory size limitations
  auto LocalMemorySize = [args] (std::vector<size_t> v) {
    return (((v[0]*v[1]*v[2]/v[3]) + (v[4]*v[5]*v[6]/v[7]))*GetBytes(args.precision));
  };
  tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"SA", "KWG", "MWG", "VWM",
                                                  "SB", "KWG", "NWG", "VWN"});

  // Modifies the thread-sizes (both global and local) based on the parameters
  tuner.MulLocalSize(id, {"MDIMC", "NDIMC"});
  tuner.MulGlobalSize(id, {"MDIMC", "NDIMC"});
  tuner.DivGlobalSize(id, {"MWG", "NWG"});

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(args.m));
  tuner.AddArgumentScalar(static_cast<int>(args.n));
  tuner.AddArgumentScalar(static_cast<int>(args.k));
  tuner.AddArgumentScalar(args.alpha);
  tuner.AddArgumentScalar(args.beta);
  tuner.AddArgumentInput(a_mat);
  tuner.AddArgumentInput(b_mat);
  tuner.AddArgumentOutput(c_mat);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void TunerXgemm(int argc, char *argv[]) {
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: TunerABC<float>(argc, argv, XgemmTune<float>); break;
    case Precision::kDouble: TunerABC<double>(argc, argv, XgemmTune<double>); break;
    case Precision::kComplexSingle: TunerABC<float2>(argc, argv, XgemmTune<float2>); break;
    case Precision::kComplexDouble: TunerABC<double2>(argc, argv, XgemmTune<double2>); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::TunerXgemm(argc, argv);
  return 0;
}

// =================================================================================================
