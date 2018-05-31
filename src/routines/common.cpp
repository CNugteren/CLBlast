
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
#include <iostream>

#include "routines/common.hpp"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(Kernel &kernel, Queue &queue, const Device &device,
               std::vector<size_t> global, const std::vector<size_t> &local,
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
      throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsTotal,
                             ToString(local_size) + " is larger than " + ToString(device.MaxWorkGroupSize()));
    }

    // Make sure the global thread sizes are at least equal to the local sizes
    for (auto i=size_t{0}; i<global.size(); ++i) {
      if (global[i] < local[i]) { global[i] = local[i]; }
    }

    // Verify that the global thread sizes are a multiple of the local sizes
    for (auto i=size_t{0}; i<global.size(); ++i) {
      if ((global[i] / local[i]) * local[i] != global[i]) {
        throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsDim,
                               ToString(global[i]) + " is not divisible by " + ToString(local[i]));
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

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(Queue &queue, const Device &device,
                const std::shared_ptr<Program> program,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t m, const size_t n, const size_t ld, const size_t offset,
                const Buffer<T> &dest, const T constant_value, const size_t local_size) {
  auto kernel = Kernel(program, "FillMatrix");
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(ld));
  kernel.SetArgument(3, static_cast<int>(offset));
  kernel.SetArgument(4, dest());
  kernel.SetArgument(5, GetRealArg(constant_value));
  auto local = std::vector<size_t>{local_size, 1};
  auto global = std::vector<size_t>{Ceil(m, local_size), n};
  RunKernel(kernel, queue, device, global, local, event, waitForEvents);
}

// Compiles the above function
template void FillMatrix<half>(Queue&, const Device&, const std::shared_ptr<Program>,
                               EventPointer, const std::vector<Event>&, const size_t, const size_t,
                               const size_t, const size_t, const Buffer<half>&, const half, const size_t);
template void FillMatrix<float>(Queue&, const Device&, const std::shared_ptr<Program>,
                                EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                const size_t, const size_t, const Buffer<float>&, const float, const size_t);
template void FillMatrix<double>(Queue&, const Device&, const std::shared_ptr<Program>,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const size_t, const Buffer<double>&, const double, const size_t);
template void FillMatrix<float2>(Queue&, const Device&, const std::shared_ptr<Program>,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const size_t, const Buffer<float2>&, const float2, const size_t);
template void FillMatrix<double2>(Queue&, const Device&, const std::shared_ptr<Program>,
                                  EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                  const size_t, const size_t, const Buffer<double2>&, const double2, const size_t);

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(Queue &queue, const Device &device,
                const std::shared_ptr<Program> program,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t n, const size_t inc, const size_t offset,
                const Buffer<T> &dest, const T constant_value, const size_t local_size) {
  auto kernel = Kernel(program, "FillVector");
  kernel.SetArgument(0, static_cast<int>(n));
  kernel.SetArgument(1, static_cast<int>(inc));
  kernel.SetArgument(2, static_cast<int>(offset));
  kernel.SetArgument(3, dest());
  kernel.SetArgument(4, GetRealArg(constant_value));
  auto local = std::vector<size_t>{local_size};
  auto global = std::vector<size_t>{Ceil(n, local_size)};
  RunKernel(kernel, queue, device, global, local, event, waitForEvents);
}

// Compiles the above function
template void FillVector<half>(Queue&, const Device&, const std::shared_ptr<Program>,
                               EventPointer, const std::vector<Event>&, const size_t, const size_t,
                               const size_t, const Buffer<half>&, const half, const size_t);
template void FillVector<float>(Queue&, const Device&, const std::shared_ptr<Program>,
                                EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                const size_t, const Buffer<float>&, const float, const size_t);
template void FillVector<double>(Queue&, const Device&, const std::shared_ptr<Program>,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const Buffer<double>&, const double, const size_t);
template void FillVector<float2>(Queue&, const Device&, const std::shared_ptr<Program>,
                                 EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                 const size_t, const Buffer<float2>&, const float2, const size_t);
template void FillVector<double2>(Queue&, const Device&, const std::shared_ptr<Program>,
                                  EventPointer, const std::vector<Event>&, const size_t, const size_t,
                                  const size_t, const Buffer<double2>&, const double2, const size_t);

// =================================================================================================
} // namespace clblast
