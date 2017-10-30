
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides helper functions for time measurement and such.
//
// =================================================================================================

#ifndef CLBLAST_TIMING_H_
#define CLBLAST_TIMING_H_

#include <cstdio>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

template <typename F>
double TimeFunction(const size_t num_runs, F const &function) {
  function(); // warm-up
  auto timings = std::vector<double>(num_runs);
  for (auto &timing: timings) {
    const auto start_time = std::chrono::steady_clock::now();
    function();
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
  }
  return *std::min_element(timings.begin(), timings.end());
}

// =================================================================================================

using Timing = std::pair<size_t, double>;

template <typename T, typename F>
std::vector<Timing> TimeRoutine(const size_t from, const size_t to, const size_t step,
                                const size_t num_runs, const Queue& queue,
                                const std::vector<Buffer<T>>& buffers, F const &routine) {
  auto timings = std::vector<Timing>();
  for (auto value = from; value < to; value += step) {
    printf("[ RUN      ] Running with value %zu\n", value);
    try {
      const auto FunctionToTune = [&]() { routine(value, queue, buffers); };
      const auto time_ms = TimeFunction(num_runs, FunctionToTune);
      printf("[       OK ] Took %.2lf ms\n", time_ms);
      timings.push_back({value, time_ms});
    }
    catch (...) {
      printf("[    ERROR ] Exception caught\n");
      timings.push_back({value, -1.0}); // invalid
    }
  }
  return timings;
}

// =================================================================================================

using TuningParameter = std::pair<std::string, size_t>;
using TuningParameters = std::vector<TuningParameter>;
struct TuningResult { std::string name; double score; TuningParameters parameters; };

void PrintTimingsToFileAsJSON(const std::string &filename,
                              const Device& device, const Platform& platform,
                              const std::vector<std::pair<std::string,std::string>> &metadata,
                              const std::vector<TuningResult>& tuning_results) {
  auto file = fopen(filename.c_str(), "w");
  fprintf(file, "{\n");
  for (auto &datum: metadata) {
    fprintf(file, "  \"%s\": \"%s\",\n", datum.first.c_str(), datum.second.c_str());
  }
  fprintf(file, "  \"platform_version\": \"%s\",\n", platform.Version().c_str());
  fprintf(file, "  \"device_name\": \"%s\",\n", GetDeviceName(device).c_str());
  fprintf(file, "  \"device_vendor\": \"%s\",\n", platform.Vendor().c_str());
  fprintf(file, "  \"device_type\": \"%s\",\n", device.Type().c_str());
  fprintf(file, "  \"device_architecture\": \"%s\",\n", GetDeviceArchitecture(device).c_str());
  fprintf(file, "  \"device_core_clock\": \"%zu\",\n", device.CoreClock());
  fprintf(file, "  \"device_compute_units\": \"%zu\",\n", device.ComputeUnits());
  fprintf(file, "  \"results\": [\n");

  // Loops over all results
  auto num_results = tuning_results.size();
  for (auto r = size_t{0}; r < num_results; ++r) {
    auto result = tuning_results[r];
    fprintf(file, "    {\n");
    fprintf(file, "      \"kernel\": \"%s\",\n", result.name.c_str());
    fprintf(file, "      \"time\": %.3lf,\n", result.score);

    // Loops over all the parameters for this result
    fprintf(file, "      \"parameters\": {");
    auto num_configs = result.parameters.size();
    for (auto p=size_t{0}; p<num_configs; ++p) {
      auto config = result.parameters[p];
      fprintf(file, "\"%s\": %zu", config.first.c_str(), config.second);
      if (p < num_configs-1) { fprintf(file, ","); }
    }
    fprintf(file, "}\n");

    // The footer
    fprintf(file, "    }");
    if (r < num_results - 1) { fprintf(file, ","); }
    fprintf(file, "\n");
  }
  fprintf(file, "  ]\n");
  fprintf(file, "}\n");
  fclose(file);
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TIMING_H_
#endif
