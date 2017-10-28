
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the interface to the CLTune auto-tuner. This is only used for the optional
// and stand-alone tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_H_
#define CLBLAST_TUNING_H_

#include <vector>
#include <string>
#include <random>
#include <utility>

#include <cltune.h>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Structures for the tuners with all the default settings
struct TunerDefaults {

  // The list of arguments relevant for this routine
  std::vector<std::string> options = {};

  // Default sizes
  size_t default_m = 1;
  size_t default_n = 1;
  size_t default_k = 1;

  // Other defaults
  size_t default_batch_count = 1;
  size_t default_num_runs = 10; // run every kernel this many times for averaging

  // Search heuristic defaults
  double default_fraction = 1.0;
  size_t default_swarm_size_PSO = 8;
  double default_influence_global_PSO = 0.1;
  double default_influence_local_PSO = 0.3;
  double default_influence_random_PSO = 0.6;
  size_t default_heuristic = static_cast<size_t>(cltune::SearchMethod::FullSearch);
  double default_max_temp_ann = 1.0;
};

// Structures for the tuners with the remaining settings
struct TunerSettings {

  // The representative kernel and the source code
  std::string kernel_family;
  std::string kernel_name;
  std::string sources;

  // Describes how to obtain the sizes of the buffers
  size_t size_x = 1;
  size_t size_y = 1;
  size_t size_a = 1;
  size_t size_b = 1;
  size_t size_c = 1;
  size_t size_temp = 1;

  // Sets the base thread configuration
  std::vector<size_t> global_size = {};
  std::vector<size_t> global_size_ref = {};
  std::vector<size_t> local_size = {};
  std::vector<size_t> local_size_ref = {};

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  TransformVector mul_local = {};
  TransformVector div_local = {};
  TransformVector mul_global = {};
  TransformVector div_global = {};

  // Sets the tuning parameters and their possible values
  std::vector<std::pair<std::string, std::vector<size_t>>> parameters;

  // Describes how to compute the performance metrics
  size_t metric_amount = 0;
  std::string performance_unit = "N/A";

  // Returns which search heuristic to use
  size_t heuristic = static_cast<size_t>(cltune::SearchMethod::FullSearch);
};

// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for all types of kernel families. Note that this is a header-only function so
// that it is automatically compiled for the various kernels (given as the 'C' template argument).
template <typename C, typename T>
void Tuner(int argc, char* argv[]) {
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  // Sets the parameters and platform/device for which to tune (command-line options)
  const TunerDefaults defaults = C::GetTunerDefaults();
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  args.device_id   = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  args.precision   = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  for (auto &o: defaults.options) {
    if (o == kArgM)        { args.m        = GetArgument(command_line_args, help, kArgM, defaults.default_m); }
    if (o == kArgN)        { args.n        = GetArgument(command_line_args, help, kArgN, defaults.default_n); }
    if (o == kArgK)        { args.k        = GetArgument(command_line_args, help, kArgK, defaults.default_k); }
    if (o == kArgAlpha)    { args.alpha    = GetArgument(command_line_args, help, kArgAlpha, GetScalar<T>()); }
    if (o == kArgBeta)     { args.beta     = GetArgument(command_line_args, help, kArgBeta, GetScalar<T>()); }
    if (o == kArgFraction) { args.fraction = GetArgument(command_line_args, help, kArgFraction, defaults.default_fraction); }
    if (o == kArgBatchCount) { args.batch_count = GetArgument(command_line_args, help, kArgBatchCount, defaults.default_batch_count); }
    if (o == kArgHeuristicSelection) {args.heuristic_selection = GetArgument(command_line_args, help, kArgHeuristicSelection, defaults.default_heuristic);  }
    if (o == kArgPsoSwarmSize)   {args.pso_swarm_size      = GetArgument(command_line_args, help, kArgPsoSwarmSize , defaults.default_swarm_size_PSO);  }
    if (o == kArgPsoInfGlobal)   {args.pso_inf_global      = GetArgument(command_line_args, help, kArgPsoInfGlobal, defaults.default_influence_global_PSO);  }
    if (o == kArgPsoInfLocal)    {args.pso_inf_local       = GetArgument(command_line_args, help, kArgPsoInfLocal, defaults.default_influence_local_PSO);  }
    if (o == kArgPsoInfRandom)   {args.pso_inf_random      = GetArgument(command_line_args, help, kArgPsoInfRandom, defaults.default_influence_random_PSO);  }
    if (o == kArgAnnMaxTemp)     {args.ann_max_temperature = GetArgument(command_line_args, help, kArgAnnMaxTemp, defaults.default_max_temp_ann); }
  }
  const auto num_runs = GetArgument(command_line_args, help, kArgNumRuns, defaults.default_num_runs);
  fprintf(stdout, "%s\n", help.c_str());
  const TunerSettings settings = C::GetTunerSettings(args);

  // Tests validity of the given arguments
  C::TestValidArguments(args);

  // Tests for validity of the precision and retrieves properties
  auto isAMD = false;
  auto isARM = false;
  auto isGPU = false;
  auto device_type = std::string{};
  auto device_vendor = std::string{};
  auto device_architecture = std::string{};
  auto device_name = std::string{};
  { // In a block such that the platform and the device are destroyed before initializing the tuner
    const auto platform = Platform(args.platform_id);
    const auto device = Device(platform, args.device_id);
    if (!PrecisionSupported<T>(device)) {
      printf("* Unsupported precision, skipping this tuning run\n\n");
      return;
    }
    isAMD = device.IsAMD();
    isARM = device.IsARM();
    isGPU = device.IsGPU();
    device_type = GetDeviceType(device);
    device_vendor = GetDeviceVendor(device);
    device_architecture = GetDeviceArchitecture(device);
    device_name = GetDeviceName(device);
  }

  // Creates input buffers with random data
  auto x_vec = std::vector<T>(settings.size_x);
  auto y_vec = std::vector<T>(settings.size_y);
  auto a_mat = std::vector<T>(settings.size_a);
  auto b_mat = std::vector<T>(settings.size_b);
  auto c_mat = std::vector<T>(settings.size_c);
  auto temp = std::vector<T>(settings.size_temp);
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  PopulateVector(x_vec, mt, dist);
  PopulateVector(y_vec, mt, dist);
  PopulateVector(a_mat, mt, dist);
  PopulateVector(b_mat, mt, dist);
  PopulateVector(c_mat, mt, dist);
  PopulateVector(temp, mt, dist);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Select the search method based on the command-line arguments
  // If the tuner does not support the selected choice, full search will be returned.
  auto method = settings.heuristic;
  if      (method == 1) { tuner.UseRandomSearch(1.0/args.fraction); }
  else if (method == 2) { tuner.UseAnnealing(1.0/args.fraction, args.ann_max_temperature); }
  else if (method == 3) { tuner.UsePSO(1.0/args.fraction, args.pso_swarm_size, args.pso_inf_global,
                                       args.pso_inf_local, args.pso_inf_random); }
  else                  { tuner.UseFullSearch(); }

  // Set extra settings for specific defines. This mimics src/routine.cc.
  auto defines = std::string{""};
  if (isAMD && isGPU) {
    defines += "#define USE_CL_MAD 1\n";
    defines += "#define USE_STAGGERED_INDICES 1\n";
  }
  if (isARM && isGPU) {
    defines += "#define GLOBAL_MEM_FENCE 1\n";
  }

  // Loads the kernel sources and defines the kernel to tune
  auto sources = defines + settings.sources;
  auto id = tuner.AddKernelFromString(sources, settings.kernel_name, settings.global_size, settings.local_size);
  tuner.SetReferenceFromString(sources, settings.kernel_name, settings.global_size_ref, settings.local_size_ref);

  // Sets the tunable parameters and their possible values
  for (const auto &parameter: settings.parameters) {
    tuner.AddParameter(id, parameter.first, parameter.second);
  }
  C::SetConstraints(tuner, id);
  C::SetLocalMemorySize(tuner, id, args);

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Modifies the thread-sizes (both global and local) based on the parameters
  for (auto &parameters: settings.mul_local) { tuner.MulLocalSize(id, parameters); }
  for (auto &parameters: settings.div_local) { tuner.DivLocalSize(id, parameters); }
  for (auto &parameters: settings.mul_global) { tuner.MulGlobalSize(id, parameters); }
  for (auto &parameters: settings.div_global) { tuner.DivGlobalSize(id, parameters); }

  // Sets the function's arguments
  C::SetArguments(tuner, args, x_vec, y_vec, a_mat, b_mat, c_mat, temp);

  // Starts the tuning process
  tuner.SetNumRuns(num_runs);
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GB/s or GFLOPS
  if (time_ms != 0.0) {
    printf("[ -------> ] %.2lf ms", time_ms);
    printf(" or %.1lf %s\n", settings.metric_amount/(time_ms*1.0e6), settings.performance_unit.c_str());
  }

  // Outputs the results as JSON to disk, including some meta-data
  auto precision_string = std::to_string(static_cast<size_t>(args.precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
    {"kernel_family", settings.kernel_family},
    {"precision", precision_string},
    {"clblast_device_type", device_type},
    {"clblast_device_vendor", device_vendor},
    {"clblast_device_architecture", device_architecture},
    {"clblast_device_name", device_name}
  };
  for (auto &o: defaults.options) {
    if (o == kArgM)     { metadata.push_back({"arg_m", std::to_string(args.m)}); }
    if (o == kArgN)     { metadata.push_back({"arg_n", std::to_string(args.n)}); }
    if (o == kArgK)     { metadata.push_back({"arg_k", std::to_string(args.k)}); }
    if (o == kArgAlpha) { metadata.push_back({"arg_alpha", ToString(args.alpha)}); }
    if (o == kArgBeta)  { metadata.push_back({"arg_beta", ToString(args.beta)}); }
    if (o == kArgBatchCount) { metadata.push_back({"arg_batch_count", ToString(args.batch_count)}); }
  }
  tuner.PrintJSON("clblast_" + settings.kernel_family + "_" + precision_string + ".json", metadata);
 
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_H_
#endif
