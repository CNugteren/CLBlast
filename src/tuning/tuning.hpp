
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

#include <cltune.h>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for all types of kernel families. Note that this is a header-only function so
// that it is automatically compiled for the various kernels (given as the 'C' template argument).
template <typename C, typename T>
void Tuner(int argc, char* argv[]) {
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  args.device_id   = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  args.precision   = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  for (auto &o: C::GetOptions()) {
    if (o == kArgM)        { args.m        = GetArgument(command_line_args, help, kArgM, C::DefaultM()); }
    if (o == kArgN)        { args.n        = GetArgument(command_line_args, help, kArgN, C::DefaultN()); }
    if (o == kArgK)        { args.k        = GetArgument(command_line_args, help, kArgK, C::DefaultK()); }
    if (o == kArgAlpha)    { args.alpha    = GetArgument(command_line_args, help, kArgAlpha, GetScalar<T>()); }
    if (o == kArgBeta)     { args.beta     = GetArgument(command_line_args, help, kArgBeta, GetScalar<T>()); }
    if (o == kArgFraction) { args.fraction = GetArgument(command_line_args, help, kArgFraction, C::DefaultFraction()); }
    if (o == kArgBatchCount) { args.batch_count = GetArgument(command_line_args, help, kArgBatchCount, C::DefaultBatchCount()); }
    if (o == kArgHeuristicSelection) {args.heuristic_selection = GetArgument(command_line_args, help, kArgHeuristicSelection, C::DefaultHeuristic());  }
    if (o == kArgPsoSwarmSize)   {args.pso_swarm_size      = GetArgument(command_line_args, help, kArgPsoSwarmSize , C::DefaultSwarmSizePSO());  }
    if (o == kArgPsoInfGlobal)   {args.pso_inf_global      = GetArgument(command_line_args, help, kArgPsoInfGlobal, C::DefaultInfluenceGlobalPSO());  }
    if (o == kArgPsoInfLocal)    {args.pso_inf_local       = GetArgument(command_line_args, help, kArgPsoInfLocal, C::DefaultInfluenceLocalPSO());  }
    if (o == kArgPsoInfRandom)   {args.pso_inf_random      = GetArgument(command_line_args, help, kArgPsoInfRandom, C::DefaultInfluenceRandomPSO());  }
    if (o == kArgAnnMaxTemp)     {args.ann_max_temperature = GetArgument(command_line_args, help, kArgAnnMaxTemp, C::DefaultMaxTempAnn());}
  }
  const auto num_runs = GetArgument(command_line_args, help, kArgNumRuns, C::DefaultNumRuns());

  fprintf(stdout, "%s\n", help.c_str());

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
  auto x_vec = std::vector<T>(C::GetSizeX(args));
  auto y_vec = std::vector<T>(C::GetSizeY(args));
  auto a_mat = std::vector<T>(C::GetSizeA(args));
  auto b_mat = std::vector<T>(C::GetSizeB(args));
  auto c_mat = std::vector<T>(C::GetSizeC(args));
  auto temp = std::vector<T>(C::GetSizeTemp(args));
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

  // Select the search method based on the cmd_line arguments
  // If the tuner does not support the selected choice, Full Search will be returned.
  auto method = C::GetHeuristic(args);
  
  if      (method == 1) { tuner.UseRandomSearch(1.0/args.fraction); }
  else if (method == 2) { tuner.UseAnnealing(1.0/args.fraction, args.ann_max_temperature); }
  else if (method == 3) { 
    tuner.UsePSO(1.0/args.fraction, args.pso_swarm_size, args.pso_inf_global, args.pso_inf_local, args.pso_inf_random);
  }
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
  auto sources = defines + C::GetSources();
  auto id = tuner.AddKernelFromString(sources, C::KernelName(), C::GlobalSize(args), C::LocalSize());
  tuner.SetReferenceFromString(sources, C::KernelName(), C::GlobalSizeRef(args), C::LocalSizeRef());

  // Sets the tunable parameters and their possible values
  C::SetParameters(tuner, id);
  C::SetConstraints(tuner, id);
  C::SetLocalMemorySize(tuner, id, args);

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Modifies the thread-sizes (both global and local) based on the parameters
  for (auto &parameters: C::MulLocal()) { tuner.MulLocalSize(id, parameters); }
  for (auto &parameters: C::DivLocal()) { tuner.DivLocalSize(id, parameters); }
  for (auto &parameters: C::MulGlobal()) { tuner.MulGlobalSize(id, parameters); }
  for (auto &parameters: C::DivGlobal()) { tuner.DivGlobalSize(id, parameters); }

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
    printf(" or %.1lf %s\n", C::GetMetric(args)/(time_ms*1.0e6), C::PerformanceUnit().c_str());
  }

  // Outputs the results as JSON to disk, including some meta-data
  auto precision_string = std::to_string(static_cast<size_t>(args.precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
    {"kernel_family", C::KernelFamily()},
    {"precision", precision_string},
    {"clblast_device_type", device_type},
    {"clblast_device_vendor", device_vendor},
    {"clblast_device_architecture", device_architecture},
    {"clblast_device_name", device_name}
  };
  for (auto &o: C::GetOptions()) {
    if (o == kArgM)     { metadata.push_back({"arg_m", std::to_string(args.m)}); }
    if (o == kArgN)     { metadata.push_back({"arg_n", std::to_string(args.n)}); }
    if (o == kArgK)     { metadata.push_back({"arg_k", std::to_string(args.k)}); }
    if (o == kArgAlpha) { metadata.push_back({"arg_alpha", ToString(args.alpha)}); }
    if (o == kArgBeta)  { metadata.push_back({"arg_beta", ToString(args.beta)}); }
    if (o == kArgBatchCount) { metadata.push_back({"arg_batch_count", ToString(args.batch_count)}); }
  }
  tuner.PrintJSON("clblast_"+C::KernelFamily()+"_"+precision_string+".json", metadata);
 
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_H_
#endif
