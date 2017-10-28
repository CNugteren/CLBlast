
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

#include <vector>
#include <chrono>

namespace clblast {
// =================================================================================================

template <typename F>
double TimeFunction(const size_t num_runs, F const &function) {
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
} // namespace clblast

// CLBLAST_TIMING_H_
#endif
