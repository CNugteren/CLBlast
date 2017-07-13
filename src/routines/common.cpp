
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

template <typename T>
void PadCopyTransposeMatrix(Queue &queue, const Device &device,
                            const Databases &db,
                            EventPointer event, const std::vector<Event> &waitForEvents,
                            const size_t src_one, const size_t src_two,
                            const size_t src_ld, const size_t src_offset,
                            const Buffer<T> &src,
                            const size_t dest_one, const size_t dest_two,
                            const size_t dest_ld, const size_t dest_offset,
                            const Buffer<T> &dest,
                            const T alpha,
                            const Program &program, const bool do_pad,
                            const bool do_transpose, const bool do_conjugate,
                            const bool upper = false, const bool lower = false,
                            const bool diagonal_imag_zero = false) {

  // Determines whether or not the fast-version could potentially be used
  auto use_fast_kernel = (src_offset == 0) && (dest_offset == 0) && (do_conjugate == false) &&
                         (src_one == dest_one) && (src_two == dest_two) && (src_ld == dest_ld) &&
                         (upper == false) && (lower == false) && (diagonal_imag_zero == false);

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db["TRA_WPT"]) &&
        IsMultiple(src_one, db["TRA_WPT"]*db["TRA_DIM"]) &&
        IsMultiple(src_two, db["TRA_WPT"]*db["TRA_DIM"])) {
      kernel_name = "TransposeMatrixFast";
    }
    else {
      use_fast_kernel = false;
      kernel_name = (do_pad) ? "TransposePadMatrix" : "TransposeMatrix";
    }
  }
  else {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db["COPY_VW"]) &&
        IsMultiple(src_one, db["COPY_VW"]*db["COPY_DIMX"]) &&
        IsMultiple(src_two, db["COPY_WPT"]*db["COPY_DIMY"])) {
      kernel_name = "CopyMatrixFast";
    }
    else {
      use_fast_kernel = false;
      kernel_name = (do_pad) ? "CopyPadMatrix" : "CopyMatrix";
    }
  }

  // Retrieves the kernel from the compiled binary
  auto kernel = Kernel(program, kernel_name);

  // Sets the kernel arguments
  if (use_fast_kernel) {
    kernel.SetArgument(0, static_cast<int>(src_ld));
    kernel.SetArgument(1, src());
    kernel.SetArgument(2, dest());
    kernel.SetArgument(3, GetRealArg(alpha));
  }
  else {
    kernel.SetArgument(0, static_cast<int>(src_one));
    kernel.SetArgument(1, static_cast<int>(src_two));
    kernel.SetArgument(2, static_cast<int>(src_ld));
    kernel.SetArgument(3, static_cast<int>(src_offset));
    kernel.SetArgument(4, src());
    kernel.SetArgument(5, static_cast<int>(dest_one));
    kernel.SetArgument(6, static_cast<int>(dest_two));
    kernel.SetArgument(7, static_cast<int>(dest_ld));
    kernel.SetArgument(8, static_cast<int>(dest_offset));
    kernel.SetArgument(9, dest());
    kernel.SetArgument(10, GetRealArg(alpha));
    if (do_pad) {
      kernel.SetArgument(11, static_cast<int>(do_conjugate));
    }
    else {
      kernel.SetArgument(11, static_cast<int>(upper));
      kernel.SetArgument(12, static_cast<int>(lower));
      kernel.SetArgument(13, static_cast<int>(diagonal_imag_zero));
    }
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    if (use_fast_kernel) {
      const auto global = std::vector<size_t>{
              dest_one / db["TRA_WPT"],
              dest_two / db["TRA_WPT"]
      };
      const auto local = std::vector<size_t>{db["TRA_DIM"], db["TRA_DIM"]};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
    else {
      const auto global = std::vector<size_t>{
              Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
              Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"])
      };
      const auto local = std::vector<size_t>{db["PADTRA_TILE"], db["PADTRA_TILE"]};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
  }
  else {
    if (use_fast_kernel) {
      const auto global = std::vector<size_t>{
              dest_one / db["COPY_VW"],
              dest_two / db["COPY_WPT"]
      };
      const auto local = std::vector<size_t>{db["COPY_DIMX"], db["COPY_DIMY"]};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
    else {
      const auto global = std::vector<size_t>{
              Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
              Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"])
      };
      const auto local = std::vector<size_t>{db["PAD_DIMX"], db["PAD_DIMY"]};
      RunKernel(kernel, queue, device, global, local, event, waitForEvents);
    }
  }
}

// Compiles the above code
template void PadCopyTransposeMatrix<half>(
        Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<half> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<half> &,
        const Program &, const bool, const bool, const bool);
template void PadCopyTransposeMatrix<float>(
        Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float> &,
        const Program &, const bool, const bool, const bool);
template void PadCopyTransposeMatrix<double>(
        Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double> &,
        const Program &, const bool, const bool, const bool);
template void PadCopyTransposeMatrix<float2>(
        Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float2> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float2> &,
        const Program &, const bool, const bool, const bool);
template void PadCopyTransposeMatrix<double2>(
        Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double2> &,
        const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double2> &,
        const Program &, const bool, const bool, const bool);

// =================================================================================================

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixBatched(Queue &queue, const Device &device,
                                   const Databases &db,
                                   EventPointer event, const std::vector<Event> &waitForEvents,
                                   const size_t src_one, const size_t src_two,
                                   const size_t src_ld, const Buffer<int> &src_offsets,
                                   const Buffer<T> &src,
                                   const size_t dest_one, const size_t dest_two,
                                   const size_t dest_ld, const Buffer<int> &dest_offsets,
                                   const Buffer<T> &dest,
                                   const Program &program, const bool do_pad,
                                   const bool do_transpose, const bool do_conjugate,
                                   const size_t batch_count) {

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    kernel_name = (do_pad) ? "TransposePadMatrixBatched" : "TransposeMatrixBatched";
  }
  else {
    kernel_name = (do_pad) ? "CopyPadMatrixBatched" : "CopyMatrixBatched";
  }

  // Retrieves the kernel from the compiled binary
  auto kernel = Kernel(program, kernel_name);

  // Sets the kernel arguments
  kernel.SetArgument(0, static_cast<int>(src_one));
  kernel.SetArgument(1, static_cast<int>(src_two));
  kernel.SetArgument(2, static_cast<int>(src_ld));
  kernel.SetArgument(3, src_offsets());
  kernel.SetArgument(4, src());
  kernel.SetArgument(5, static_cast<int>(dest_one));
  kernel.SetArgument(6, static_cast<int>(dest_two));
  kernel.SetArgument(7, static_cast<int>(dest_ld));
  kernel.SetArgument(8, dest_offsets());
  kernel.SetArgument(9, dest());
  if (do_pad) {
    kernel.SetArgument(10, static_cast<int>(do_conjugate));
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    const auto global = std::vector<size_t>{
            Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
            Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
            batch_count
    };
    const auto local = std::vector<size_t>{db["PADTRA_TILE"], db["PADTRA_TILE"], 1};
    RunKernel(kernel, queue, device, global, local, event, waitForEvents);
  }
  else {
    const auto global = std::vector<size_t>{
            Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
            Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"]),
            batch_count
    };
    const auto local = std::vector<size_t>{db["PAD_DIMX"], db["PAD_DIMY"], 1};
    RunKernel(kernel, queue, device, global, local, event, waitForEvents);
  }
}

// Compiles the above code
template void PadCopyTransposeMatrixBatched<half>(
    Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<half> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<half> &,
    const Program &, const bool, const bool, const bool, const size_t);
template void PadCopyTransposeMatrixBatched<float>(
    Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float> &,
    const Program &, const bool, const bool, const bool, const size_t);
template void PadCopyTransposeMatrixBatched<double>(
    Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double> &,
    const Program &, const bool, const bool, const bool, const size_t);
template void PadCopyTransposeMatrixBatched<float2>(
    Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float2> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<float2> &,
    const Program &, const bool, const bool, const bool, const size_t);
template void PadCopyTransposeMatrixBatched<double2>(
    Queue &, const Device &, const Databases &, EventPointer, const std::vector<Event> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double2> &,
    const size_t, const size_t, const size_t, const Buffer<int> &, const Buffer<double2> &,
    const Program &, const bool, const bool, const bool, const size_t);

// =================================================================================================
} // namespace clblast
