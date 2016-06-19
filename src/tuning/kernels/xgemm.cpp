
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xgemm OpenCL kernels.
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
class TuneXgemm {
 public:

  // The representative kernel and the source code
  static std::string KernelFamily() { return "xgemm"; }
  static std::string KernelName() { return "Xgemm"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level3/xgemm_part1.opencl"
      #include "../src/kernels/level3/xgemm_part2.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgM, kArgN, kArgK, kArgAlpha, kArgBeta, kArgFraction};
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 1024; }
  static size_t DefaultN() { return 1024; }
  static size_t DefaultK() { return 1024; }
  static double DefaultFraction() { return 2048.0; }

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeY(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeA(const Arguments<T> &args) { return args.m * args.k; }
  static size_t GetSizeB(const Arguments<T> &args) { return args.n * args.k; }
  static size_t GetSizeC(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
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
  }

  // Sets the constraints
  static void SetConstraints(cltune::Tuner &tuner, const size_t id) {
    auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
    auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
    auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };
    // Requirement for unrolling the KWG loop
    tuner.AddConstraint(id, MultipleOfX, {"KWG", "KWI"});
    // Required for integer MWI and NWI
    tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMC", "VWM"});
    tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMC", "VWN"});
    // Required for integer MWIA and NWIB
    tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMA", "VWM"});
    tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMB", "VWN"});
    // KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
    tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "MDIMA"});
    tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "NDIMB"});
  }

  // Sets the local memory size
  static void SetLocalMemorySize(cltune::Tuner &tuner, const size_t id, const Arguments<T> &args) {
    auto LocalMemorySize = [args] (std::vector<size_t> v) {
      return (((v[0]*v[1]*v[2]/v[3]) + (v[4]*v[5]*v[6]/v[7]))*GetBytes(args.precision));
    };
    tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"SA", "KWG", "MWG", "VWM",
                                                    "SB", "KWG", "NWG", "VWN"});
  }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &args) { return {args.m, args.n}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &args) { return GlobalSize(args); }
  static std::vector<size_t> LocalSize() { return {1, 1}; }
  static std::vector<size_t> LocalSizeRef() { return {8, 8}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"MDIMC", "NDIMC"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {{"MDIMC", "NDIMC"}}; }
  static TransformVector DivGlobal() { return {{"MWG", "NWG"}}; }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &, std::vector<T> &,
                           std::vector<T> &a_mat, std::vector<T> &b_mat, std::vector<T> &c_mat,
                           std::vector<T> &) {
    auto alpha_buffer = std::vector<T>{args.alpha};
    auto beta_buffer = std::vector<T>{args.beta};
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(static_cast<int>(args.k));
    tuner.AddArgumentInput(alpha_buffer);
    tuner.AddArgumentInput(beta_buffer);
    tuner.AddArgumentInput(a_mat);
    tuner.AddArgumentInput(b_mat);
    tuner.AddArgumentOutput(c_mat);
  }

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return 2 * args.m * args.n * args.k;
  }
  static std::string PerformanceUnit() { return "GFLOPS"; }
};

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  switch(clblast::GetPrecision(argc, argv)) {
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXgemm<half>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXgemm<float>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXgemm<double>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXgemm<float2>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXgemm<double2>, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
