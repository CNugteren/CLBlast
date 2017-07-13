
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

#include "clpp11.hpp"
#include "clblast.h"
#include "database/database.hpp"

namespace clblast {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(const Kernel &kernel, const Queue &queue, const Device &device,
               const std::vector<size_t> &global, const std::vector<size_t> &local,
               EventPointer event, const std::vector<Event> &waitForEvents = {});

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(Queue &queue, const Device &device,
                const Program &program, const Databases &,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t m, const size_t n, const size_t ld, const size_t offset,
                const Buffer<T> &dest,
                const T constant_value) {
  auto kernel = Kernel(program, "FillMatrix");
  kernel.SetArgument(0, static_cast<int>(m));
  kernel.SetArgument(1, static_cast<int>(n));
  kernel.SetArgument(2, static_cast<int>(ld));
  kernel.SetArgument(3, static_cast<int>(offset));
  kernel.SetArgument(4, dest());
  kernel.SetArgument(5, GetRealArg(constant_value));
  auto local = std::vector<size_t>{8, 8};
  auto global = std::vector<size_t>{Ceil(m, 8), Ceil(n, 8)};
  RunKernel(kernel, queue, device, global, local, event, waitForEvents);
}

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(Queue &queue, const Device &device,
                const Program &program, const Databases &,
                EventPointer event, const std::vector<Event> &waitForEvents,
                const size_t n, const size_t inc, const size_t offset,
                const Buffer<T> &dest,
                const T constant_value) {
  auto kernel = Kernel(program, "FillVector");
  kernel.SetArgument(0, static_cast<int>(n));
  kernel.SetArgument(1, static_cast<int>(inc));
  kernel.SetArgument(2, static_cast<int>(offset));
  kernel.SetArgument(3, dest());
  kernel.SetArgument(4, GetRealArg(constant_value));
  auto local = std::vector<size_t>{64};
  auto global = std::vector<size_t>{Ceil(n, 64)};
  RunKernel(kernel, queue, device, global, local, event, waitForEvents);
}

// =================================================================================================

// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
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
                            const bool diagonal_imag_zero = false);

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
                                   const size_t batch_count);

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_COMMON_H_
#endif
