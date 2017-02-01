
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common routine functions (see the header for more information).
//
// =================================================================================================

#include <vector>
#include <chrono>

#include "routines/common.hpp"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(const Kernel &kernel, const Queue &queue, const Device &device,
               const std::vector<size_t> &global, const std::vector<size_t> &local,
               EventPointer event, const std::vector<Event> &waitForEvents) {

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
      if (global[i] < local[i]) {
        throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsTotal);
      }
    }
  }

  // Tests for local memory usage
  const auto local_mem_usage = kernel.LocalMemUsage(device);
  if (!device.IsLocalMemoryValid(local_mem_usage)) {
    throw RuntimeErrorCode(StatusCode::kInvalidLocalMemUsage);
  }

  // Prints the name of the kernel to launch in case of debugging in verbose mode
  #ifdef VERBOSE
    queue.Finish();
    printf("[DEBUG] Running kernel '%s'\n", kernel.GetFunctionName().c_str());
    const auto start_time = std::chrono::steady_clock::now();
  #endif

  // Launches the kernel (and checks for launch errors)
  kernel.Launch(queue, global, local, event, waitForEvents);

  // Prints the elapsed execution time in case of debugging in verbose mode
  #ifdef VERBOSE
    queue.Finish();
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
    printf("[DEBUG] Completed kernel in %.2lf ms\n", timing);
  #endif
}

// =================================================================================================
} // namespace clblast
