
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the interfaces to common kernels, such as copying, padding, and
// transposing a matrix. These functions are templated and thus header-only. This file also contains
// other common functions to routines, such as a function to launch a kernel.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_COMMON_H_
#define CLBLAST_ROUTINES_COMMON_H_

#include <string>
#include <vector>

#include "clblast.h"
#include "clpp11.hpp"
#include "database/database.hpp"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
StatusCode RunKernel(Kernel &kernel, Queue &queue, const Device &device,
                     std::vector<size_t> global, const std::vector<size_t> &local,
                     EventPointer event, const std::vector<Event> &waitForEvents = {});

// =================================================================================================

// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
template <typename T>
StatusCode PadCopyTransposeMatrix(Queue &queue, const Device &device,
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
        IsMultiple(src_one, db["TRA_WPT"]*db["TRA_WPT"]) &&
        IsMultiple(src_two, db["TRA_WPT"]*db["TRA_WPT"])) {
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
  try {
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
        return RunKernel(kernel, queue, device, global, local, event, waitForEvents);
      }
      else {
        const auto global = std::vector<size_t>{
          Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
          Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"])
        };
        const auto local = std::vector<size_t>{db["PADTRA_TILE"], db["PADTRA_TILE"]};
        return RunKernel(kernel, queue, device, global, local, event, waitForEvents);
      }
    }
    else {
      if (use_fast_kernel) {
        const auto global = std::vector<size_t>{
          dest_one / db["COPY_VW"],
          dest_two / db["COPY_WPT"]
        };
        const auto local = std::vector<size_t>{db["COPY_DIMX"], db["COPY_DIMY"]};
        return RunKernel(kernel, queue, device, global, local, event, waitForEvents);
      }
      else {
        const auto global = std::vector<size_t>{
          Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
          Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"])
        };
        const auto local = std::vector<size_t>{db["PAD_DIMX"], db["PAD_DIMY"]};
        return RunKernel(kernel, queue, device, global, local, event, waitForEvents);
      }
    }
  } catch (...) { return StatusCode::kInvalidKernel; }
}

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_COMMON_H_
#endif
