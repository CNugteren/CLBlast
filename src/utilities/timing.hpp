
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

void PrintTimingsToFileAsJSON(const std::string &filename,
                              const Device& device, const Platform& platform,
                              const std::vector<std::pair<std::string,std::string>> &descriptions) {
  auto file = fopen(filename.c_str(), "w");
  fprintf(file, "{\n");
  for (auto &description: descriptions) {
    fprintf(file, "  \"%s\": \"%s\",\n", description.first.c_str(), description.second.c_str());
  }
  fprintf(file, "  \"platform_version\": \"%s\",\n", platform.Version().c_str());
  fprintf(file, "  \"device_name\": \"%s\",\n", GetDeviceName(device).c_str());
  fprintf(file, "  \"device_vendor\": \"%s\",\n", platform.Vendor().c_str());
  fprintf(file, "  \"device_type\": \"%s\",\n", device.Type().c_str());
  fprintf(file, "  \"device_architecture\": \"%s\",\n", GetDeviceArchitecture(device).c_str());
  fprintf(file, "  \"device_core_clock\": \"%zu\",\n", device.CoreClock());
  fprintf(file, "  \"device_compute_units\": \"%zu\",\n", device.ComputeUnits());
  fprintf(file, "}\n");
  fclose(file);
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TIMING_H_
#endif
