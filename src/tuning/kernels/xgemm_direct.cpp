
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the direct xgemm kernels. There are two variations:
// - V==1: This tests some limited set of tuning parameters exhaustively.
// - V==2: This tests a much larger set of tuning parameters by randomly sampling a subset.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, int V>
class TuneXgemmDirect {
 public:

  // The representative kernel and the source code
  static std::string KernelFamily() { return (V==1) ? "xgemm_direct_1" : "xgemm_direct_2"; }
  static std::string KernelName() { return "XgemmDirect"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level3/xgemm_direct_part1.opencl"
      #include "../src/kernels/level3/xgemm_direct_part2.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgM, kArgN, kArgK, kArgAlpha, kArgBeta, kArgFraction};
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 256; }
  static size_t DefaultN() { return 256; }
  static size_t DefaultK() { return 256; }
  static double DefaultFraction() { return (V==1) ? 1.0 : 16.0; } // test all or sample randomly
  static size_t DefaultNumRuns() { return 10; } // run every kernel this many times for averaging

  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeY(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeA(const Arguments<T> &args) { return args.m * args.k; }
  static size_t GetSizeB(const Arguments<T> &args) { return args.n * args.k; }
  static size_t GetSizeC(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    if (V==1) { // limited subset of tuning parameters - but explorable exhaustively
      tuner.AddParameter(id, "WGD", {8, 16, 32});
      tuner.AddParameter(id, "MDIMCD", {8, 16, 32});
      tuner.AddParameter(id, "NDIMCD", {8, 16, 32});
      tuner.AddParameter(id, "MDIMAD", {8, 16, 32});
      tuner.AddParameter(id, "NDIMBD", {8, 16, 32});
      tuner.AddParameter(id, "KWID", {2});
      tuner.AddParameter(id, "VWMD", {1, 2, 4, 8});
      tuner.AddParameter(id, "VWND", {1, 2, 4, 8});
      tuner.AddParameter(id, "PADA", {1});
      tuner.AddParameter(id, "PADB", {1});
    } // a lot more tuning parameters - has to be sampled randomly, too much to test all
    else {
      tuner.AddParameter(id, "WGD", {8, 16, 32, 64, 128});
      tuner.AddParameter(id, "MDIMCD", {8, 16, 32});
      tuner.AddParameter(id, "NDIMCD", {8, 16, 32});
      tuner.AddParameter(id, "MDIMAD", {8, 16, 32});
      tuner.AddParameter(id, "NDIMBD", {8, 16, 32});
      tuner.AddParameter(id, "KWID", {2, 8, 16});
      tuner.AddParameter(id, "VWMD", {1, 2, 4, 8});
      tuner.AddParameter(id, "VWND", {1, 2, 4, 8});
      tuner.AddParameter(id, "PADA", {0, 1});
      tuner.AddParameter(id, "PADB", {0, 1});
    }
  }

  // Sets the constraints
  static void SetConstraints(cltune::Tuner &tuner, const size_t id) {
    auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
    auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
    auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };
    // Requirement for unrolling the WGD loop
    tuner.AddConstraint(id, MultipleOfX, {"WGD", "KWID"});
    // Required for integer MWID and NWID
    tuner.AddConstraint(id, MultipleOfXMulY, {"WGD", "MDIMCD", "VWMD"});
    tuner.AddConstraint(id, MultipleOfXMulY, {"WGD", "NDIMCD", "VWND"});
    // Required for integer MWIAD and NWIBD
    tuner.AddConstraint(id, MultipleOfXMulY, {"WGD", "MDIMAD", "VWMD"});
    tuner.AddConstraint(id, MultipleOfXMulY, {"WGD", "NDIMBD", "VWND"});
    // WGD has to be a multiple of KDIMAD = ((MDIMCD*NDIMCD)/(MDIMAD)) and KDIMBD = (...)
    tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"WGD", "MDIMCD", "NDIMCD", "MDIMAD"});
    tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"WGD", "MDIMCD", "NDIMCD", "NDIMBD"});

    // Extra constraints for variation 1 to limit the set of options significantly
    if (V==1) {
      auto IsEqual = [] (std::vector<size_t> v) { return v[0] == v[1]; };
      tuner.AddConstraint(id, IsEqual, {"MDIMCD", "MDIMAD"});
      tuner.AddConstraint(id, IsEqual, {"NDIMCD", "NDIMBD"});
    }
  }

  // Sets the local memory size
  static void SetLocalMemorySize(cltune::Tuner &tuner, const size_t id, const Arguments<T> &args) {
    auto LocalMemorySize = [args] (std::vector<size_t> v) {
      return ((v[0]*(v[0] + v[1]) + v[0]*(v[0] + v[2]))*GetBytes(args.precision));
    };
    tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"WGD", "PADA", "PADB"});
  }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &args) { return {args.m, args.n}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &args) { return GlobalSize(args); }
  static std::vector<size_t> LocalSize() { return {1, 1}; }
  static std::vector<size_t> LocalSizeRef() { return {8, 8}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"MDIMCD", "NDIMCD"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {{"MDIMCD", "NDIMCD"}}; }
  static TransformVector DivGlobal() { return {{"WGD", "WGD"}}; }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &, std::vector<T> &,
                           std::vector<T> &a_mat, std::vector<T> &b_mat, std::vector<T> &c_mat,
                           std::vector<T> &) {
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(static_cast<int>(args.k));
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentScalar(GetRealArg(args.beta));
    tuner.AddArgumentInput(a_mat);
    tuner.AddArgumentScalar(0); // a_offset
    tuner.AddArgumentScalar(static_cast<int>(args.k)); // a_ld
    tuner.AddArgumentInput(b_mat);
    tuner.AddArgumentScalar(0); // b_offset
    tuner.AddArgumentScalar(static_cast<int>(args.n)); // b_ld
    tuner.AddArgumentOutput(c_mat);
    tuner.AddArgumentScalar(0); // c_offset
    tuner.AddArgumentScalar(static_cast<int>(args.n)); // c_ld
    tuner.AddArgumentScalar(1); // a_do_transpose
    tuner.AddArgumentScalar(0); // b_do_transpose
    tuner.AddArgumentScalar(1); // c_do_transpose
    tuner.AddArgumentScalar(0); // a_conjugate
    tuner.AddArgumentScalar(0); // b_conjugate
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

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  switch(clblast::GetPrecision(argc, argv)) {
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXgemmDirect<half,V>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXgemmDirect<float,V>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXgemmDirect<double,V>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXgemmDirect<float2,V>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXgemmDirect<double2,V>, double2>(argc, argv); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  return 0;
}

// =================================================================================================
