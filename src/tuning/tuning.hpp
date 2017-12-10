
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the generic CLBlast auto-tuner (inspired by CLTune). This is only used for
//  the optional and stand-alone tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_TUNING_H_
#define CLBLAST_TUNING_TUNING_H_

#include <vector>
#include <string>
#include <random>
#include <utility>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "utilities/utilities.hpp"
#include "utilities/compile.hpp"
#include "utilities/timing.hpp"
#include "tuning/configurations.hpp"

namespace clblast {
// =================================================================================================

// Constants holding start and end strings for terminal-output in colour
#if defined(_WIN32)
  const std::string kPrintError = "";
  const std::string kPrintSuccess = "";
  const std::string kPrintMessage = "";
  const std::string kPrintEnd = "";
#else
  const std::string kPrintError = "\x1b[31m";
  const std::string kPrintSuccess = "\x1b[32m";
  const std::string kPrintMessage = "\x1b[1m";
  const std::string kPrintEnd = "\x1b[0m";
#endif

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
  double default_fraction = 1.0;
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

  // Inputs and outputs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  std::vector<size_t> inputs = {};
  std::vector<size_t> outputs = {};

  // Sets the base thread configuration
  std::vector<size_t> global_size = {};
  std::vector<size_t> global_size_ref = {};
  std::vector<size_t> local_size = {};
  std::vector<size_t> local_size_ref = {};

  // Transforms the thread configuration based on the parameters
  TransformVector mul_local = {};
  TransformVector div_local = {};
  TransformVector mul_global = {};
  TransformVector div_global = {};

  // Sets the tuning parameters and their possible values
  std::vector<Parameter> parameters;

  // Describes how to compute the performance metrics
  size_t metric_amount = 0;
  std::string performance_unit = "N/A";
};

// =================================================================================================

struct TuningResult { std::string name; double score; Configuration config; };

void PrintTimingsToFileAsJSON(const std::string &filename,
                              const Device& device, const Platform& platform,
                              const std::vector<std::pair<std::string,std::string>> &metadata,
                              const std::vector<TuningResult>& tuning_results);

void print_separator(const size_t parameters_size);

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
    if (o == kArgBatchCount) { args.batch_count = GetArgument(command_line_args, help, kArgBatchCount, defaults.default_batch_count); }
  }
  args.fraction = GetArgument(command_line_args, help, kArgFraction, defaults.default_fraction);
  args.num_runs = GetArgument(command_line_args, help, kArgNumRuns, defaults.default_num_runs);
  const auto max_l2_norm = GetArgument(command_line_args, help, kArgMaxL2Norm, 1.0e-4);
  printf("%s\n", help.c_str());
  const TunerSettings settings = C::GetTunerSettings(args);

  // Tests validity of the given arguments
  C::TestValidArguments(args);

  // Initializes OpenCL
  const auto platform = Platform(args.platform_id);
  const auto device = Device(platform, args.device_id);
  const auto context = Context(device);

  // Tests for validity of the precision and retrieves properties
  if (!PrecisionSupported<T>(device)) {
    printf("* Unsupported precision, skipping this tuning run\n\n");
    return;
  }
  const auto device_type = GetDeviceType(device);
  const auto device_vendor = GetDeviceVendor(device);
  const auto device_architecture = GetDeviceArchitecture(device);
  const auto device_name = GetDeviceName(device);

  // Creates input buffers with random data
  const auto buffer_sizes = std::vector<size_t>{
      settings.size_x, settings.size_y,
      settings.size_a, settings.size_b, settings.size_c,
      settings.size_temp
  };
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  auto source_buffers = std::vector<std::vector<T>>();
  auto reference_buffers = std::vector<std::vector<T>>();
  auto result_buffers = std::vector<std::vector<T>>();
  auto device_buffers = std::vector<Buffer<T>>();
  for (const auto size : buffer_sizes) {
    auto host_buffer = std::vector<T>(size);
    PopulateVector(host_buffer, mt, dist);
    source_buffers.push_back(host_buffer);
    reference_buffers.push_back(std::vector<T>(size));
    result_buffers.push_back(std::vector<T>(size));
    device_buffers.push_back(Buffer<T>(context, size));
  }

  // Sets the tunable parameters and their possible values
  auto configurations = SetConfigurations(settings.parameters, C::SetConstraints());
  printf("* Found %s%zu configuration(s)%s\n",
         kPrintMessage.c_str(), configurations.size(), kPrintEnd.c_str());

  // Select the search method (full search or a random fraction)
  if (args.fraction != 0.0 && args.fraction != 1.0) {
    const auto new_size = static_cast<size_t>(configurations.size() / args.fraction);
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(configurations), std::end(configurations), rng);
    configurations.resize(new_size);
    printf("* Exploring a random subset of %s%zu configuration(s)%s\n",
           kPrintMessage.c_str(), configurations.size(), kPrintEnd.c_str());
  }

  // Prints information about the parameters
  printf("* Parameters explored: ");
  for (const auto& parameter : settings.parameters) { printf("%s ", parameter.first.c_str()); }
  printf("\n");

  // Prints the header of the table
  printf("\n");
  printf("|   ID | total |");
  for (auto i = size_t{0}; i < settings.parameters.size() - 1; ++i) { printf("     "); }
  printf("param |       compiles |         time | %6s |            status |\n", settings.performance_unit.c_str());
  print_separator(settings.parameters.size());

  // First runs a reference example to compare against
  try {
    auto queue = Queue(context, device);
    printf("|  ref |     - |");
    for (auto i = size_t{0}; i < settings.parameters.size() - 1; ++i) { printf("     "); }
    printf("    - |");


    // Sets the input
    for (const auto id : settings.inputs) {
      device_buffers[id].Write(queue, buffer_sizes[id], source_buffers[id]);
    }

    // Compiles the kernel
    auto compiler_options = std::vector<std::string>();
    const auto program = CompileFromSource(settings.sources, args.precision, settings.kernel_name,
                                           device, context, compiler_options, 0);
    auto kernel = Kernel(program, settings.kernel_name);
    C::SetArguments(kernel, args, device_buffers);
    printf("             %sOK%s |", kPrintSuccess.c_str(), kPrintEnd.c_str());

    // Runs the kernel
    const auto time_ms = TimeKernel(args.num_runs, kernel, queue, device,
                                    settings.global_size_ref, settings.local_size_ref);
    printf("      - |");
    if (time_ms == -1.0) { throw std::runtime_error("Error in reference implementation"); }

    // Saves the result
    for (const auto id : settings.outputs) {
      device_buffers[id].Read(queue, buffer_sizes[id], reference_buffers[id]);
    }
    printf("      %sreference OK%s |\n", kPrintSuccess.c_str(), kPrintEnd.c_str());
  }
  catch (...) {
    const auto status_code = DispatchExceptionCatchAll(true);
    printf("* Exception caught with status %d while running the reference, aborting\n",
           static_cast<int>(status_code));
    return;
  }
  print_separator(settings.parameters.size());

  // Starts the tuning process
  auto results = std::vector<TuningResult>();
  for (auto config_id = size_t{0}; config_id < configurations.size(); ++config_id) {
    try {
      auto queue = Queue(context, device);

      auto configuration = configurations[config_id];
      printf("| %4zu | %5zu |", config_id + 1, configurations.size());
      for (const auto& parameter : settings.parameters) {
        printf("%5zu", configuration.at(parameter.first));
      }
      printf(" |");

      // Sets the input
      for (const auto id : settings.inputs) {
        device_buffers[id].Write(queue, buffer_sizes[id], source_buffers[id]);
      }

      // Sets the thread configuration
      const auto global = SetThreadConfiguration(configuration, settings.global_size,
                                                 settings.mul_global, settings.div_global);
      const auto local = SetThreadConfiguration(configuration, settings.local_size,
                                                settings.mul_local, settings.div_local);

      // Sets the parameters for this configuration
      auto kernel_source = std::string{""};
      for (const auto &parameter : configuration) {
        kernel_source += "#define " + parameter.first + " " + ToString(parameter.second) + "\n";
      }
      kernel_source += settings.sources;

      // Compiles the kernel
      const auto start_time = std::chrono::steady_clock::now();
      auto compiler_options = std::vector<std::string>();
      const auto program = CompileFromSource(kernel_source, args.precision, settings.kernel_name,
                                             device, context, compiler_options, 0, true);
      auto kernel = Kernel(program, settings.kernel_name);
      const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
      const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
      printf("   %sOK%s  %5.0lf ms |", kPrintSuccess.c_str(), kPrintEnd.c_str(), timing);

      // Runs the kernel
      C::SetArguments(kernel, args, device_buffers);
      const auto time_ms = TimeKernel(args.num_runs, kernel, queue, device, global, local);

      // Kernel run was not successful
      if (time_ms == -1.0) {
        printf("      - |");
        printf("   %sinvalid config.%s |", kPrintError.c_str(), kPrintEnd.c_str());
        printf(" <-- skipping\n");
        continue;
      }

      // Compares the results
      auto l2_error = 0.0;
      for (const auto id : settings.outputs) {
        device_buffers[id].Read(queue, buffer_sizes[id], result_buffers[id]);
        for (auto index = size_t{0}; index<buffer_sizes[id]; ++index) {
          const auto diff = SquaredDifference(result_buffers[id][index], reference_buffers[id][index]);
          l2_error += diff;
        }
        l2_error /= static_cast<double>(buffer_sizes[id]);
        if (std::isnan(l2_error) || l2_error > max_l2_norm) {
          printf("      - |");
          printf(" %sL2 error %8.2e%s |", kPrintError.c_str(), l2_error, kPrintEnd.c_str());
          throw std::runtime_error("L2 error too large");
        }
      }

      // All was OK
      configuration["PRECISION"] = static_cast<size_t>(args.precision);
      results.push_back(TuningResult{settings.kernel_name, time_ms, configuration});
      printf(" %6.1lf |", settings.metric_amount / (time_ms * 1.0e6));
      printf("     %sresults match%s |\n", kPrintSuccess.c_str(), kPrintEnd.c_str());
    }
    catch (CLCudaAPIBuildError) {
      const auto status_code = DispatchExceptionCatchAll(true);
      printf("  %scompilation error: %5d%s     |",
             kPrintError.c_str(), static_cast<int>(status_code), kPrintEnd.c_str());
      printf("      - |                 - | <-- skipping\n");
    }
    catch (...) {
      const auto status_code = DispatchExceptionCatchAll(true);
      if (status_code != StatusCode::kUnknownError) {
        printf("   %serror code %d%s |",
               kPrintError.c_str(), static_cast<int>(status_code), kPrintEnd.c_str());
      }
      printf(" <-- skipping\n");
    }
  }

  // Completed the tuning process
  print_separator(settings.parameters.size());
  printf("\n");
  if (results.size() == 0) { return; }

  // Computes the best results
  auto comparison = [](const TuningResult& lhs, const TuningResult& rhs) { return lhs.score < rhs.score; };
  const auto best_configuration = std::min_element(results.begin(), results.end(), comparison);
  const auto best_time_ms = best_configuration->score;
  if (best_time_ms == 0.0) { return; }

  // Also prints the performance of the best-case in terms of GB/s or GFLOPS
  printf("\n");
  printf("* Found best result %.2lf ms", best_time_ms);
  printf(": %.1lf %s\n", settings.metric_amount / (best_time_ms * 1.0e6),
         settings.performance_unit.c_str());
  printf("* Best parameters: ");
  auto best_string = std::string{""};
  auto i = size_t{0};
  for (const auto config : best_configuration->config) {
    best_string += "" + config.first + "=" + ToString(config.second);
    if (i < best_configuration->config.size() - 1) { best_string += " "; }
    ++i;
  }
  printf("%s\n\n", best_string.c_str());

  // Outputs the results as JSON to disk, including some meta-data
  auto precision_string = std::to_string(static_cast<size_t>(args.precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
    {"kernel_family", settings.kernel_family},
    {"precision", precision_string},
    {"best_kernel", best_configuration->name},
    {"best_time", ToString(best_configuration->score)},
    {"best_parameters", best_string}
  };
  for (auto &o: defaults.options) {
    if (o == kArgM)     { metadata.push_back({"arg_m", ToString(args.m)}); }
    if (o == kArgN)     { metadata.push_back({"arg_n", ToString(args.n)}); }
    if (o == kArgK)     { metadata.push_back({"arg_k", ToString(args.k)}); }
    if (o == kArgAlpha) { metadata.push_back({"arg_alpha", ToString(args.alpha)}); }
    if (o == kArgBeta)  { metadata.push_back({"arg_beta", ToString(args.beta)}); }
    if (o == kArgBatchCount) { metadata.push_back({"arg_batch_count", ToString(args.batch_count)}); }
  }
  PrintTimingsToFileAsJSON("clblast_" + settings.kernel_family + "_" + precision_string + ".json",
                           device, platform, metadata, results);

  printf("* Completed tuning process\n");
  printf("\n");
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_TUNING_H_
#endif
