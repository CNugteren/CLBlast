
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xgemm OpenCL kernels. There are two variations:
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

// Settings for this kernel (default command-line arguments)
TunerDefaults XgemmGetTunerDefaults(const int V) {
  auto settings = TunerDefaults();
  settings.options = {kArgM, kArgN, kArgK, kArgAlpha, kArgBeta, kArgFraction,
                      kArgHeuristicSelection, kArgPsoSwarmSize,
                      kArgPsoInfGlobal, kArgPsoInfLocal, kArgPsoInfRandom};
  settings.default_m = 1024;
  settings.default_n = 1024;
  settings.default_k = 1024;
  settings.default_fraction = (V==1) ? 1.0 : 512.0; // test all or sample randomly
  settings.default_num_runs = 2;
  return settings;
}

// Settings for this kernel (general)
template <typename T>
TunerSettings XgemmGetTunerSettings(const int V, const Arguments<T> &args) {
  auto settings = TunerSettings();

  // Identification of the kernel
  settings.kernel_family = (V==1) ? "xgemm_1" : "xgemm_2";
  settings.kernel_name = "Xgemm";
  settings.sources =
#include "../src/kernels/level3/xgemm_part1.opencl"
#include "../src/kernels/level3/xgemm_part2.opencl"
#include "../src/kernels/level3/xgemm_part3.opencl"
#include "../src/kernels/level3/xgemm_part4.opencl"
  ;

  // Buffer sizes
  settings.size_a = args.m * args.k;
  settings.size_b = args.n * args.k;
  settings.size_c = args.m * args.n;

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {2, 3, 4};
  settings.outputs = {4};

  // Sets the base thread configuration
  settings.global_size = {args.m, args.n};
  settings.global_size_ref = {args.m / 4, args.n / 8};
  settings.local_size = {1, 1};
  settings.local_size_ref = {2, 1};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"MDIMC", "NDIMC"}};
  settings.mul_global = {{"MDIMC", "NDIMC"}};
  settings.div_global = {{"MWG", "NWG"}};

  // Sets the tuning parameters and their possible values
  settings.parameters = {
      {"MWG", {8, 16, 32, 64, 128, 256}},
      {"NWG", {8, 16, 32, 64, 128, 256}},
      {"KWG", {4}},
      {"MDIMC", {2, 4, 8, 16, 32}},
      {"NDIMC", {2, 4, 8, 16, 32}},
      {"MDIMA", {2, 4, 8, 16, 32}},
      {"NDIMB", {2, 4, 8, 16, 32}},
      {"KWI", {1}},
      {"VWM", {4}},
      {"VWN", {4}},
      {"STRM", {0}},
      {"STRN", {0}},
      {"SA", {0}},
      {"SB", {0}},
  };

  // Describes how to compute the performance metrics
  settings.metric_amount = 2 * args.m * args.n * args.k;
  settings.performance_unit = "GFLOPS";

  return settings;
}

// Tests for valid arguments
template <typename T>
void XgemmTestValidArguments(const int, const Arguments<T> &) { }
std::vector<Constraint> XgemmSetConstraints(const int V) {
  auto constraints = std::vector<Constraint>();
  auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
  auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
  auto EqualOfXMulY = [] (std::vector<size_t> v) { return v[0] == v[1]*v[2]; };
  // Required for matching MWI and VWN/VWM
  constraints.push_back({EqualOfXMulY, {"MWG", "MDIMC", "VWM"}});
  constraints.push_back({EqualOfXMulY, {"MWG", "MDIMC", "VWN"}});
  // Required for integer MWIA and NWIB
  constraints.push_back({MultipleOfXMulY, {"MWG", "MDIMA", "VWM"}});
  constraints.push_back({MultipleOfXMulY, {"NWG", "NDIMB", "VWN"}});

  // Extra constraints for variation 1 to limit the set of options significantly
  if (V==1) {
    auto IsEqual = [] (std::vector<size_t> v) { return v[0] == v[1]; };
    constraints.push_back({IsEqual, {"MDIMC", "MDIMA"}});
    constraints.push_back({IsEqual, {"NDIMC", "NDIMB"}});
    constraints.push_back({IsEqual, {"SA", "SB"}});
  }
  return constraints;
}

// Sets the kernel's arguments
template <typename T>
void XgemmSetArguments(const int, Kernel &kernel, const Arguments<T> &args, std::vector<Buffer<T>>& buffers) {
  kernel.SetArgument(0, static_cast<int>(args.m));
  kernel.SetArgument(1, static_cast<int>(args.n));
  kernel.SetArgument(2, static_cast<int>(args.k));
  kernel.SetArgument(3, GetRealArg(args.alpha));
  kernel.SetArgument(4, GetRealArg(args.beta));
  kernel.SetArgument(5, buffers[2]()); // 2 == A matrix
  kernel.SetArgument(6, buffers[3]()); // 3 == B matrix
  kernel.SetArgument(7, buffers[4]()); // 4 == C matrix
  kernel.SetArgument(8, 0);
  kernel.SetArgument(9, 0);
}

// =================================================================================================
} // namespace clblast
