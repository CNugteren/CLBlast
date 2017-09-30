
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

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, int V>
class TuneXgemmDirect {
 public:

  // Settings for this kernel (default command-line arguments)
  static TunerDefaults GetTunerDefaults() {
    auto settings = TunerDefaults();
    settings.options = {kArgM, kArgN, kArgK, kArgAlpha, kArgBeta, kArgFraction,
                        kArgHeuristicSelection, kArgPsoSwarmSize,
                        kArgPsoInfGlobal, kArgPsoInfLocal, kArgPsoInfRandom};
    settings.default_m = 256;
    settings.default_n = 256;
    settings.default_k = 256;
    settings.default_fraction = (V==1) ? 1.0 : 32.0; // test all or sample randomly
    settings.default_num_runs = 4;
    settings.default_heuristic = static_cast<size_t>(cltune::SearchMethod::RandomSearch);
    return settings;
  }

  // Settings for this kernel (general)
  static TunerSettings GetTunerSettings(const Arguments<T> &args) {
    auto settings = TunerSettings();

    // Identification of the kernel
    settings.kernel_family = (V==1) ? "xgemm_direct_1" : "xgemm_direct_2";
    settings.kernel_name = "XgemmDirectTN";
    settings.sources =
#include "../src/kernels/common.opencl"
#include "../src/kernels/level3/xgemm_direct_part1.opencl"
#include "../src/kernels/level3/xgemm_direct_part2.opencl"
#include "../src/kernels/level3/xgemm_direct_part3.opencl"
    ;

    // Buffer sizes
    settings.size_a = args.m * args.k;
    settings.size_b = args.n * args.k;
    settings.size_c = args.m * args.n;

    // Sets the base thread configuration
    settings.global_size = {args.m, args.n};
    settings.global_size_ref = settings.global_size;
    settings.local_size = {1, 1};
    settings.local_size_ref = {8, 8};

    // Transforms the thread configuration based on the parameters
    settings.mul_local = {{"MDIMCD", "NDIMCD"}};
    settings.mul_global = {{"MDIMCD", "NDIMCD"}};
    settings.div_global = {{"WGD", "WGD"}};

    // Sets the tuning parameters and their possible values
    if (V==1) { // limited subset of tuning parameters - but explorable exhaustively
      settings.parameters = {
        {"WGD", {8, 16, 32}},
        {"MDIMCD", {8, 16, 32}},
        {"NDIMCD", {8, 16, 32}},
        {"MDIMAD", {8, 16, 32}},
        {"NDIMBD", {8, 16, 32}},
        {"KWID", {2}},
        {"VWMD", {1, 2, 4, 8}},
        {"VWND", {1, 2, 4, 8}},
        {"PADA", {1}},
        {"PADB", {1}},
      };
    }
    else { // a lot more tuning parameters - has to be sampled randomly, too much to test all
      settings.parameters = {
        {"WGD", {8, 16, 32, 64, 128}},
        {"MDIMCD", {8, 16, 32}},
        {"NDIMCD", {8, 16, 32}},
        {"MDIMAD", {8, 16, 32}},
        {"NDIMBD", {8, 16, 32}},
        {"KWID", {2, 8, 16}},
        {"VWMD", {1, 2, 4, 8}},
        {"VWND", {1, 2, 4, 8}},
        {"PADA", {0, 1}},
        {"PADB", {0, 1}},
      };
    }

    // Describes how to compute the performance metrics
    settings.metric_amount = 2 * args.m * args.n * args.k;
    settings.performance_unit = "GFLOPS";

    // Returns which search heuristic to use
    if (V==1) { settings.heuristic = static_cast<size_t>(cltune::SearchMethod::FullSearch); }
    else {
      // Use full-search to explore all parameter combinations or another strategy to search only a
      // part of the parameter values. The fraction is set as a command-line argument.
      if (args.fraction == 1.0 || args.fraction == 0.0) {
        settings.heuristic = static_cast<size_t>(cltune::SearchMethod::FullSearch);
      } else {
        settings.heuristic = args.heuristic_selection;
      }
    }

    return settings;
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

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
    tuner.AddArgumentScalar(1); // c_do_transpose
    tuner.AddArgumentScalar(0); // a_conjugate
    tuner.AddArgumentScalar(0); // b_conjugate
  }
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
