
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
                            const Database &db,
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
                            const bool upper, const bool lower,
                            const bool diagonal_imag_zero) {

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

template
void PadCopyTransposeMatrix<half>(Queue &queue, const Device &device,
                                  const Database &db,
                                  EventPointer event, const std::vector<Event> &waitForEvents,
                                  const size_t src_one, const size_t src_two,
                                  const size_t src_ld, const size_t src_offset,
                                  const Buffer<half> &src,
                                  const size_t dest_one, const size_t dest_two,
                                  const size_t dest_ld, const size_t dest_offset,
                                  const Buffer<half> &dest,
                                  const half alpha,
                                  const Program &program, const bool do_pad,
                                  const bool do_transpose, const bool do_conjugate,
                                  const bool upper, const bool lower,
                                  const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<float>(Queue &queue, const Device &device,
                                   const Database &db,
                                   EventPointer event, const std::vector<Event> &waitForEvents,
                                   const size_t src_one, const size_t src_two,
                                   const size_t src_ld, const size_t src_offset,
                                   const Buffer<float> &src,
                                   const size_t dest_one, const size_t dest_two,
                                   const size_t dest_ld, const size_t dest_offset,
                                   const Buffer<float> &dest,
                                   const float alpha,
                                   const Program &program, const bool do_pad,
                                   const bool do_transpose, const bool do_conjugate,
                                   const bool upper, const bool lower,
                                   const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<float2>(Queue &queue, const Device &device,
                                    const Database &db,
                                    EventPointer event, const std::vector<Event> &waitForEvents,
                                    const size_t src_one, const size_t src_two,
                                    const size_t src_ld, const size_t src_offset,
                                    const Buffer<float2> &src,
                                    const size_t dest_one, const size_t dest_two,
                                    const size_t dest_ld, const size_t dest_offset,
                                    const Buffer<float2> &dest,
                                    const float2 alpha,
                                    const Program &program, const bool do_pad,
                                    const bool do_transpose, const bool do_conjugate,
                                    const bool upper, const bool lower,
                                    const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<double>(Queue &queue, const Device &device,
                                    const Database &db,
                                    EventPointer event, const std::vector<Event> &waitForEvents,
                                    const size_t src_one, const size_t src_two,
                                    const size_t src_ld, const size_t src_offset,
                                    const Buffer<double> &src,
                                    const size_t dest_one, const size_t dest_two,
                                    const size_t dest_ld, const size_t dest_offset,
                                    const Buffer<double> &dest,
                                    const double alpha,
                                    const Program &program, const bool do_pad,
                                    const bool do_transpose, const bool do_conjugate,
                                    const bool upper, const bool lower,
                                    const bool diagonal_imag_zero);

template
void PadCopyTransposeMatrix<double2>(Queue &queue, const Device &device,
                                     const Database &db,
                                     EventPointer event, const std::vector<Event> &waitForEvents,
                                     const size_t src_one, const size_t src_two,
                                     const size_t src_ld, const size_t src_offset,
                                     const Buffer<double2> &src,
                                     const size_t dest_one, const size_t dest_two,
                                     const size_t dest_ld, const size_t dest_offset,
                                     const Buffer<double2> &dest,
                                     const double2 alpha,
                                     const Program &program, const bool do_pad,
                                     const bool do_transpose, const bool do_conjugate,
                                     const bool upper, const bool lower,
                                     const bool diagonal_imag_zero);

// =================================================================================================
} // namespace clblast
