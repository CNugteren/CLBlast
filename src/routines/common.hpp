
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

// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
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
                            const bool upper = false, const bool lower = false,
                            const bool diagonal_imag_zero = false);

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_COMMON_H_
#endif
