
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

#include <cstdio>
#include <exception>

#include "utilities/timing.hpp"

namespace clblast {
// =================================================================================================

double RunKernelTimed(const size_t num_runs, Kernel &kernel, Queue &queue, const Device &device,
                      std::vector<size_t> global, const std::vector<size_t> &local) {
  auto event = Event();

  if (!local.empty()) {
    // Tests for validity of the local thread sizes
    if (local.size() > device.MaxWorkItemDimensions()) {
      throw RuntimeErrorCode(StatusCode::kInvalidLocalNumDimensions);
    }
    const auto max_work_item_sizes = device.MaxWorkItemSizes();
    for (auto i=size_t{0}; i<local.size(); ++i) {
      if (local[i] > max_work_item_sizes[i]) {
        throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsDim);
      }
    }
    auto local_size = size_t{1};
    for (auto &item: local) { local_size *= item; }
    if (local_size > device.MaxWorkGroupSize()) {
      throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsTotal);
    }

    // Make sure the global thread sizes are at least equal to the local sizes
    for (auto i=size_t{0}; i<global.size(); ++i) {
      if (global[i] < local[i]) { global[i] = local[i]; }
    }
  }

  // Tests for local memory usage
  const auto local_mem_usage = kernel.LocalMemUsage(device);
  if (!device.IsLocalMemoryValid(local_mem_usage)) {
    throw RuntimeErrorCode(StatusCode::kInvalidLocalMemUsage);
  }

  // Times the kernel
  const auto run_kernel_func = [&]() {
      kernel.Launch(queue, global, local, event.pointer());
      event.WaitForCompletion();
      queue.Finish();
  };
  return TimeFunction(num_runs, run_kernel_func);
}

double TimeKernel(const size_t num_runs, Kernel &kernel, Queue &queue, const Device &device,
                  std::vector<size_t> global, const std::vector<size_t> &local,
                  const bool silent) {
  try {
    const auto time_ms = RunKernelTimed(num_runs, kernel, queue, device, global, local);
    if (!silent) { printf(" %9.2lf ms |", time_ms); }
    return time_ms;
  }
  catch (...) {
    const auto status_code = DispatchExceptionCatchAll(true);
    if (!silent) { printf("  error %-5d |", static_cast<int>(status_code)); }
    return -1.0; // invalid
  }
}

// =================================================================================================
} // namespace clblast
