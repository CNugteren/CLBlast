
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the part of the auto-tuner for tuning entire routines (i.e. switching
// between direct and in-direct GEMM kernels)
//
// =================================================================================================

#ifndef CLBLAST_TUNING_ROUTINES_ROUTINE_TUNER_H_
#define CLBLAST_TUNING_ROUTINES_ROUTINE_TUNER_H_

#include <exception>
#include <string>
#include <vector>
#include <assert.h>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
void ForceSelectIndirectFrom(const size_t minimum_size, const Device &device,
                             const std::string &name, const std::string& parameter_name) {
  const auto override_status = OverrideParameters(device(), name + "Routine", PrecisionValue<T>(),
                                                  {{parameter_name, minimum_size}});
  if (override_status != StatusCode::kSuccess) {
    throw RuntimeError("OverrideParameters failed with status " + ToString(override_status));
  }
}

template <typename T, typename F>
std::vector<TuningResult> TuneKernelSelection(const Device& device, const Context& context,
                                              Queue& queue, const Precision precision,
                                              F const &routine, const size_t num_runs,
                                              const std::string &name,
                                              const std::string& parameter_name) {

  // Values for m, n, and k
  const auto from = size_t{64};
  const auto to = size_t{2048};
  const auto step = size_t{64};

  // Buffers
  auto buffers = std::vector<Buffer<T>>{
      Buffer<T>(context, to * to),
      Buffer<T>(context, to * to),
      Buffer<T>(context, to * to)
  };

  // In-direct version
  printf("\n* Testing the in-direct %s routine for m=n=k\n", name.c_str());
  ForceSelectIndirectFrom<T>(0, device, name, parameter_name);
  const auto indirect = TimeRoutine(from, to, step, num_runs, queue, buffers, routine);

  // Direct version
  printf("\n* Testing the direct %s routine for m=n=k\n", name.c_str());
  ForceSelectIndirectFrom<T>(to + 1, device, name, parameter_name);
  const auto direct = TimeRoutine(from, to, step, num_runs, queue, buffers, routine);

  // Determining final score and best kernel selection point
  assert(indirect.size() == direct.size());
  printf("\n* Collecting results\n");
  auto ratios = std::vector<double>(indirect.size());
  for (auto i = size_t{0}; i < indirect.size(); ++i) {
    ratios[i] = indirect[i].second / direct[i].second;
  }
  auto scores = std::vector<TuningResult>(ratios.size());
  for (auto i = size_t{0}; i < scores.size(); ++i) {
    auto score = 0;
    for (auto j = size_t{0}; j < i; ++j) { score += (ratios[j] <= 1.0); }
    for (auto j = i + 1; j < ratios.size(); ++j) { score += (ratios[j] > 1.0); }
    const auto epsilon = (scores.size() - i) / 1e3; // favour later results over earlier ones
    const auto relative_score = static_cast<double>(score) / static_cast<double>(scores.size() - 1);
    auto tuning_results = Configuration();
    tuning_results[parameter_name] = indirect[i].first;
    tuning_results["PRECISION"] = static_cast<size_t>(precision);
    scores[i] = TuningResult{
        name + "_kernel_selection",
        (relative_score * relative_score) * 100 + epsilon,  // squared for proper default computation
        tuning_results
    };
  }

  // Displaying results
  printf("|         || %8s indirect || %8s direct ||          |\n", name.c_str(), name.c_str());
  printf("|   m=n=k ||   ms   |  GFLOPS  ||   ms   |  GFLOPS  ||  score   | (lowest score == best switching point)\n");
  printf("x---------xx--------x----------xx--------x----------xx----------x\n");
  for (auto i = size_t{0}; i < indirect.size(); ++i) {
    assert(indirect[i].first == direct[i].first);
    const auto value = indirect[i].first;
    if (indirect[i].second != -1 && direct[i].second != -1) {
      const auto gflops_indirect = (2 * value * value * value) / (indirect[i].second * 1.0e6);
      const auto gflops_direct = (2 * value * value * value) / (direct[i].second * 1.0e6);
      printf("| %7zu || %6.2lf | %8.1lf || %6.2lf | %8.1lf || %8.3lf |\n",
             value, indirect[i].second, gflops_indirect, direct[i].second, gflops_direct, scores[i].score);
    }
  }
  printf("x---------xx--------x----------xx--------x----------xx----------x\n");
  printf("\n");
  return scores;
}

// Computes the best switching point
TuningResult GetBestResult(const std::vector<TuningResult>& scores) {
  auto comparison = [](const TuningResult& lhs, const TuningResult& rhs) { return lhs.score < rhs.score; };
  const auto best_configuration = std::min_element(scores.begin(), scores.end(), comparison);
  return *best_configuration;
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_ROUTINES_ROUTINE_TUNER_H_
#endif
