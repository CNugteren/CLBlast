
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

#include "internal/routines/common.h"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
StatusCode RunKernel(Kernel &kernel, Queue &queue, const Device &device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event, std::vector<Event>& waitForEvents) {

  // Tests for validity of the local thread sizes
  if (local.size() > device.MaxWorkItemDimensions()) {
    return StatusCode::kInvalidLocalNumDimensions; 
  }
  const auto max_work_item_sizes = device.MaxWorkItemSizes();
  for (auto i=size_t{0}; i<local.size(); ++i) {
    if (local[i] > max_work_item_sizes[i]) { return StatusCode::kInvalidLocalThreadsDim; }
  }
  auto local_size = size_t{1};
  for (auto &item: local) { local_size *= item; }
  if (local_size > device.MaxWorkGroupSize()) { return StatusCode::kInvalidLocalThreadsTotal; }

  // Make sure the global thread sizes are at least equal to the local sizes
  for (auto i=size_t{0}; i<global.size(); ++i) {
    if (global[i] < local[i]) { global[i] = local[i]; }
  }

  // Tests for local memory usage
  const auto local_mem_usage = kernel.LocalMemUsage(device);
  if (!device.IsLocalMemoryValid(local_mem_usage)) { return StatusCode::kInvalidLocalMemUsage; }

  // Launches the kernel (and checks for launch errors)
  try {
    kernel.Launch(queue, global, local, event, waitForEvents);
  } catch (...) { return StatusCode::kKernelLaunchError; }

  // No errors, normal termination of this function
  return StatusCode::kSuccess;
}

// As above, but without an event waiting list
StatusCode RunKernel(Kernel &kernel, Queue &queue, const Device &device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event) {
  auto emptyWaitingList = std::vector<Event>();
  return RunKernel(kernel, queue, device, global, local, event, emptyWaitingList);
}

// =================================================================================================
} // namespace clblast
