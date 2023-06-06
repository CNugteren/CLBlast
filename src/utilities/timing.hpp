
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

double RunKernelTimed(const size_t num_runs, Kernel &kernel, Queue &queue, const Device &device,
                      std::vector<size_t> global, const std::vector<size_t> &local);

double TimeKernel(const size_t num_runs, Kernel &kernel, Queue &queue, const Device &device,
                  std::vector<size_t> global, const std::vector<size_t> &local,
                  const bool silent = false);

// =================================================================================================

using Timing = std::pair<size_t, double>;

template <typename T, typename F>
std::vector<Timing> TimeRoutine(const size_t from, const size_t to, const size_t step,
                                const size_t num_runs, Queue& queue,
                                const std::vector<Buffer<T>>& buffers, F const &routine) {
  auto timings = std::vector<Timing>();
  printf("|  value |         time |\n");
  printf("x--------x--------------x\n");
  for (auto value = from; value < to; value += step) {
    printf("| %6zu |", value);
    try {
      const auto FunctionToTune = [&]() { routine(value, queue, buffers); };
      const auto time_ms = TimeFunction(num_runs, FunctionToTune);
      printf(" %9.2lf ms |\n", time_ms);
      timings.push_back({value, time_ms});
    }
    catch (...) {
      const auto status_code = DispatchExceptionCatchAll(true);
      printf("  error %-5d |\n", static_cast<int>(status_code));
      timings.push_back({value, -1.0}); // invalid
    }
  }
  printf("x--------x--------------x\n");
  return timings;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TIMING_H_
#endif
