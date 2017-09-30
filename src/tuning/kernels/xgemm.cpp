
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xgemm OpenCL kernels. There are two variations:
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
class TuneXgemm {
 public:

  // Settings for this kernel (default command-line arguments)
  static TunerDefaults GetTunerDefaults() {
    auto settings = TunerDefaults();
    settings.options = {kArgM, kArgN, kArgK, kArgAlpha, kArgBeta, kArgFraction,
                        kArgHeuristicSelection, kArgPsoSwarmSize,
                        kArgPsoInfGlobal, kArgPsoInfLocal, kArgPsoInfRandom};
    settings.default_m = 1024;
    settings.default_n = 1024;
    settings.default_k = 1024;
    settings.default_fraction = (V==1) ? 1.0 : 512.0; // test all or sample randomly
    settings.default_num_runs = 2;
    settings.default_heuristic = static_cast<size_t>(cltune::SearchMethod::RandomSearch);
    return settings;
  }

  // Settings for this kernel (general)
  static TunerSettings GetTunerSettings(const Arguments<T> &args) {
    auto settings = TunerSettings();

    // Identification of the kernel
    settings.kernel_family = (V==1) ? "xgemm_1" : "xgemm_2";
    settings.kernel_name = "Xgemm";
    settings.sources =
#include "../src/kernels/common.opencl"
#include "../src/kernels/level3/xgemm_part1.opencl"
#include "../src/kernels/level3/xgemm_part2.opencl"
#include "../src/kernels/level3/xgemm_part3.opencl"
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
    settings.mul_local = {{"MDIMC", "NDIMC"}};
    settings.mul_global = {{"MDIMC", "NDIMC"}};
    settings.div_global = {{"MWG", "NWG"}};

    // Sets the tuning parameters and their possible values
    if (V==1) { // limited subset of tuning parameters - but explorable exhaustively
      settings.parameters = {
        {"MWG", {16, 32, 64}},
        {"NWG", {16, 32, 64}},
        {"KWG", {32}},
        {"MDIMC", {8, 16, 32}},
        {"NDIMC", {8, 16, 32}},
        {"MDIMA", {8, 16, 32}},
        {"NDIMB", {8, 16, 32}},
        {"KWI", {2}},
        {"VWM", {1, 2, 4}},
        {"VWN", {1, 2, 4}},
        {"STRM", {0}},
        {"STRN", {0}},
        {"SA", {0, 1}},
        {"SB", {0, 1}},
      };
    }
    else { // a lot more tuning parameters - has to be sampled randomly, too much to test all
      settings.parameters = {
        {"MWG", {16, 32, 64, 128}},
        {"NWG", {16, 32, 64, 128}},
        {"KWG", {16, 32}},
        {"MDIMC", {8, 16, 32}},
        {"NDIMC", {8, 16, 32}},
        {"MDIMA", {8, 16, 32}},
        {"NDIMB", {8, 16, 32}},
        {"KWI", {2}},
        {"VWM", {1, 2, 4, 8}},
        {"VWN", {1, 2, 4, 8}},
        {"STRM", {0, 1}},
        {"STRN", {0, 1}},
        {"SA", {0, 1}},
        {"SB", {0, 1}},
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

    // Extra constraints for variation 1 to limit the set of options significantly
    if (V==1) {
      auto IsEqual = [] (std::vector<size_t> v) { return v[0] == v[1]; };
      tuner.AddConstraint(id, IsEqual, {"MDIMC", "MDIMA"});
      tuner.AddConstraint(id, IsEqual, {"NDIMC", "NDIMB"});
      tuner.AddConstraint(id, IsEqual, {"SA", "SB"});
    }
  }

  // Sets the local memory size
  static void SetLocalMemorySize(cltune::Tuner &tuner, const size_t id, const Arguments<T> &args) {
    auto LocalMemorySize = [args] (std::vector<size_t> v) {
      return (((v[0]*v[1]*v[2]) + (v[3]*v[4]*v[5]))*GetBytes(args.precision));
    };
    tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"SA", "KWG", "MWG",
                                                    "SB", "KWG", "NWG"});
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
    tuner.AddArgumentInput(b_mat);
    tuner.AddArgumentOutput(c_mat);
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
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXgemm<half,V>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXgemm<float,V>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXgemm<double,V>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXgemm<float2,V>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXgemm<double2,V>, double2>(argc, argv); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  return 0;
}

// =================================================================================================
