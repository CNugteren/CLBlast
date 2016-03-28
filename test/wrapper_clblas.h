
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a wrapper around the clBLAS library, such that its routines can be called
// in a similar way as the CLBlast routines: using alpha and beta to determine the precision.
//
// =================================================================================================

#ifndef CLBLAST_TEST_WRAPPER_CLBLAS_H_
#define CLBLAST_TEST_WRAPPER_CLBLAS_H_

#include <clBLAS.h>

#include "internal/utilities.h"

namespace clblast {

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Forwards the clBLAS calls for SSWAP/DSWAP/CSWAP/ZSWAP
template <typename T>
clblasStatus clblasXswap(const size_t n,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXswap<float>(const size_t n,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSswap(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXswap<double>(const size_t n,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDswap(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXswap<float2>(const size_t n,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCswap(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXswap<double2>(const size_t n,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZswap(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSCAL/DSCAL/CSCAL/ZSCAL
clblasStatus clblasXscal(const size_t n,
                         const float alpha,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSscal(n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXscal(const size_t n,
                         const double alpha,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDscal(n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXscal(const size_t n,
                         const float2 alpha,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCscal(n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXscal(const size_t n,
                         const double2 alpha,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZscal(n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SCOPY/DCOPY/CCOPY/ZCOPY
template <typename T>
clblasStatus clblasXcopy(const size_t n,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXcopy<float>(const size_t n,
                                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasScopy(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXcopy<double>(const size_t n,
                                 const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDcopy(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXcopy<float2>(const size_t n,
                                 const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCcopy(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXcopy<double2>(const size_t n,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZcopy(n,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SAXPY/DAXPY/CAXPY/ZAXPY
clblasStatus clblasXaxpy(const size_t n,
                         const float alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSaxpy(n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXaxpy(const size_t n,
                         const double alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDaxpy(n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXaxpy(const size_t n,
                         const float2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCaxpy(n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXaxpy(const size_t n,
                         const double2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZaxpy(n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SDOT/DDOT
template <typename T>
clblasStatus clblasXdot(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXdot<float>(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_uint num_queues, cl_command_queue *queues,
                               cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float>(context, n);
  return clblasSdot(n,
                    dot_buffer, dot_offset,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    y_buffer, y_offset, static_cast<int>(y_inc),
                    scratch_buffer(),
                    num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXdot<double>(const size_t n,
                                cl_mem dot_buffer, const size_t dot_offset,
                                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double>(context, n);
  return clblasDdot(n,
                    dot_buffer, dot_offset,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    y_buffer, y_offset, static_cast<int>(y_inc),
                    scratch_buffer(),
                    num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CDOTU/ZDOTU
template <typename T>
clblasStatus clblasXdotu(const size_t n,
                         cl_mem dot_buffer, const size_t dot_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXdotu<float2>(const size_t n,
                                 cl_mem dot_buffer, const size_t dot_offset,
                                 const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float2>(context, n);
  return clblasCdotu(n,
                     dot_buffer, dot_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXdotu<double2>(const size_t n,
                                  cl_mem dot_buffer, const size_t dot_offset,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double2>(context, n);
  return clblasZdotu(n,
                     dot_buffer, dot_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CDOTC/ZDOTC
template <typename T>
clblasStatus clblasXdotc(const size_t n,
                         cl_mem dot_buffer, const size_t dot_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXdotc<float2>(const size_t n,
                                 cl_mem dot_buffer, const size_t dot_offset,
                                 const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float2>(context, n);
  return clblasCdotc(n,
                     dot_buffer, dot_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXdotc<double2>(const size_t n,
                                  cl_mem dot_buffer, const size_t dot_offset,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double2>(context, n);
  return clblasZdotc(n,
                     dot_buffer, dot_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SNRM2/DNRM2/ScNRM2/DzNRM2
template <typename T>
clblasStatus clblasXnrm2(const size_t n,
                         cl_mem nrm2_buffer, const size_t nrm2_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXnrm2<float>(const size_t n,
                                cl_mem nrm2_buffer, const size_t nrm2_offset,
                                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float>(context, 2*n);
  return clblasSnrm2(n,
                     nrm2_buffer, nrm2_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXnrm2<double>(const size_t n,
                                 cl_mem nrm2_buffer, const size_t nrm2_offset,
                                 const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double>(context, 2*n);
  return clblasDnrm2(n,
                     nrm2_buffer, nrm2_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXnrm2<float2>(const size_t n,
                                 cl_mem nrm2_buffer, const size_t nrm2_offset,
                                 const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float2>(context, 2*n);
  return clblasScnrm2(n,
                     nrm2_buffer, nrm2_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXnrm2<double2>(const size_t n,
                                  cl_mem nrm2_buffer, const size_t nrm2_offset,
                                  const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double2>(context, 2*n);
  return clblasDznrm2(n,
                     nrm2_buffer, nrm2_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// Forwards the clBLAS calls for SGEMV/DGEMV/CGEMV/ZGEMV
clblasStatus clblasXgemv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSgemv(layout, a_transpose,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgemv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDgemv(layout, a_transpose,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgemv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCgemv(layout, a_transpose,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_float2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgemv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZgemv(layout, a_transpose,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_double2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SGBMV/DGBMV/CGBMV/ZGBMV
clblasStatus clblasXgbmv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n, const size_t kl, const size_t ku,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSgbmv(layout, a_transpose,
                     m, n, kl, ku,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgbmv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n, const size_t kl, const size_t ku,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDgbmv(layout, a_transpose,
                     m, n, kl, ku,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgbmv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n, const size_t kl, const size_t ku,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCgbmv(layout, a_transpose,
                     m, n, kl, ku,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_float2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgbmv(const clblasOrder layout, const clblasTranspose a_transpose,
                         const size_t m, const size_t n, const size_t kl, const size_t ku,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZgbmv(layout, a_transpose,
                     m, n, kl, ku,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_double2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHEMV/ZHEMV
clblasStatus clblasXhemv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasChemv(layout, triangle,
                     n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_float2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXhemv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZhemv(layout, triangle,
                     n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_double2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHBMV/ZHBMV
clblasStatus clblasXhbmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n, const size_t k,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasChbmv(layout, triangle,
                     n, k,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_float2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXhbmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n, const size_t k,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZhbmv(layout, triangle,
                     n, k,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_double2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHPMV/ZHPMV
clblasStatus clblasXhpmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float2 alpha,
                         const cl_mem ap_buffer, const size_t ap_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasChpmv(layout, triangle,
                     n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_float2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXhpmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double2 alpha,
                         const cl_mem ap_buffer, const size_t ap_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double2 beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZhpmv(layout, triangle,
                     n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     cl_double2{{beta.real(), beta.imag()}},
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSYMV/DSYMV
clblasStatus clblasXsymv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsymv(layout, triangle,
                     n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsymv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsymv(layout, triangle,
                     n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSBMV/DSBMV
clblasStatus clblasXsbmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsbmv(layout, triangle,
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsbmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsbmv(layout, triangle,
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSPMV/DSPMV
clblasStatus clblasXspmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float alpha,
                         const cl_mem ap_buffer, const size_t ap_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const float beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSspmv(layout, triangle,
                     n,
                     alpha,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXspmv(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double alpha,
                         const cl_mem ap_buffer, const size_t ap_offset,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const double beta,
                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDspmv(layout, triangle,
                     n,
                     alpha,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     beta,
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STRMV/DTRMV/CTRMV/ZTRMV
template <typename T>
clblasStatus clblasXtrmv(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t n,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXtrmv<float>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                const size_t n,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float>(context, n);
  return clblasStrmv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtrmv<double>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double>(context, n);
  return clblasDtrmv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtrmv<float2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float2>(context, n);
  return clblasCtrmv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtrmv<double2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                  const size_t n,
                                  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double2>(context, n);
  return clblasZtrmv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STBMV/DTBMV/CTBMV/ZTBMV
template <typename T>
clblasStatus clblasXtbmv(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t n, const size_t k,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXtbmv<float>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                const size_t n, const size_t k,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float>(context, n);
  return clblasStbmv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtbmv<double>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n, const size_t k,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double>(context, n);
  return clblasDtbmv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtbmv<float2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n, const size_t k,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float2>(context, n);
  return clblasCtbmv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtbmv<double2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                  const size_t n, const size_t k,
                                  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double2>(context, n);
  return clblasZtbmv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STPMV/DTPMV/CTPMV/ZTPMV
template <typename T>
clblasStatus clblasXtpmv(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t n,
                         const cl_mem ap_buffer, const size_t ap_offset,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXtpmv<float>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                const size_t n,
                                const cl_mem ap_buffer, const size_t ap_offset,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float>(context, n);
  return clblasStpmv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtpmv<double>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem ap_buffer, const size_t ap_offset,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double>(context, n);
  return clblasDtpmv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtpmv<float2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem ap_buffer, const size_t ap_offset,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<float2>(context, n);
  return clblasCtpmv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtpmv<double2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                  const size_t n,
                                  const cl_mem ap_buffer, const size_t ap_offset,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  auto queue = Queue(queues[0]);
  auto context = queue.GetContext();
  auto scratch_buffer = Buffer<double2>(context, n);
  return clblasZtpmv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     scratch_buffer(),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
clblasStatus clblasXtrsv(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t n,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXtrsv<float>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                const size_t n,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasStrsv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtrsv<double>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDtrsv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtrsv<float2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCtrsv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtrsv<double2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                  const size_t n,
                                  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZtrsv(layout, triangle, a_transpose, diagonal,
                     n,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
clblasStatus clblasXtbsv(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t n, const size_t k,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXtbsv<float>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                const size_t n, const size_t k,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasStbsv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtbsv<double>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n, const size_t k,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDtbsv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtbsv<float2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n, const size_t k,
                                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCtbsv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtbsv<double2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                  const size_t n, const size_t k,
                                  const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZtbsv(layout, triangle, a_transpose, diagonal,
                     n, k,
                     a_buffer, a_offset, a_ld,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
clblasStatus clblasXtpsv(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t n,
                         const cl_mem ap_buffer, const size_t ap_offset,
                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events);
template <>
clblasStatus clblasXtpsv<float>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                const size_t n,
                                const cl_mem ap_buffer, const size_t ap_offset,
                                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                cl_uint num_queues, cl_command_queue *queues,
                                cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasStpsv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtpsv<double>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem ap_buffer, const size_t ap_offset,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDtpsv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtpsv<float2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                 const size_t n,
                                 const cl_mem ap_buffer, const size_t ap_offset,
                                 cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                 cl_uint num_queues, cl_command_queue *queues,
                                 cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCtpsv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}
template <>
clblasStatus clblasXtpsv<double2>(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                                  const size_t n,
                                  const cl_mem ap_buffer, const size_t ap_offset,
                                  cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                  cl_uint num_queues, cl_command_queue *queues,
                                  cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZtpsv(layout, triangle, a_transpose, diagonal,
                     n,
                     ap_buffer, ap_offset,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SGER/DGER
clblasStatus clblasXger(const clblasOrder layout,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSger(layout,
                    m, n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    y_buffer, y_offset, static_cast<int>(y_inc),
                    a_buffer, a_offset, a_ld,
                    num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXger(const clblasOrder layout,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDger(layout,
                    m, n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    y_buffer, y_offset, static_cast<int>(y_inc),
                    a_buffer, a_offset, a_ld,
                    num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CGERU/ZGERU
clblasStatus clblasXgeru(const clblasOrder layout,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCgeru(layout,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgeru(const clblasOrder layout,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZgeru(layout,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CGERC/ZGERC
clblasStatus clblasXgerc(const clblasOrder layout,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCgerc(layout,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgerc(const clblasOrder layout,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZgerc(layout,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHER/ZHER
clblasStatus clblasXher(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCher(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    a_buffer, a_offset, a_ld,
                    num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXher(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZher(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    a_buffer, a_offset, a_ld,
                    num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHPR/ZHPR
clblasStatus clblasXhpr(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasChpr(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    ap_buffer, ap_offset,
                    num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXhpr(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZhpr(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    ap_buffer, ap_offset,
                    num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHER2/ZHER2
clblasStatus clblasXher2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCher2(layout, triangle,
                     n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXher2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZher2(layout, triangle,
                     n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHPR2/ZHPR2
clblasStatus clblasXhpr2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem ap_buffer, const size_t ap_offset,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasChpr2(layout, triangle,
                     n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     ap_buffer, ap_offset,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXhpr2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double2 alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem ap_buffer, const size_t ap_offset,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZhpr2(layout, triangle,
                     n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     ap_buffer, ap_offset,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSYR/DSYR
clblasStatus clblasXsyr(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsyr(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    a_buffer, a_offset, a_ld,
                    num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyr(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsyr(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    a_buffer, a_offset, a_ld,
                    num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSPR/DSPR
clblasStatus clblasXspr(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSspr(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    ap_buffer, ap_offset,
                    num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXspr(const clblasOrder layout, const clblasUplo triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_uint num_queues, cl_command_queue *queues,
                        cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDspr(layout, triangle,
                    n,
                    alpha,
                    x_buffer, x_offset, static_cast<int>(x_inc),
                    ap_buffer, ap_offset,
                    num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSYR2/DSYR2
clblasStatus clblasXsyr2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsyr2(layout, triangle,
                     n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyr2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsyr2(layout, triangle,
                     n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     a_buffer, a_offset, a_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSPR2/DSPR2
clblasStatus clblasXspr2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const float alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem ap_buffer, const size_t ap_offset,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSspr2(layout, triangle,
                     n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     ap_buffer, ap_offset,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXspr2(const clblasOrder layout, const clblasUplo triangle,
                         const size_t n,
                         const double alpha,
                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                         cl_mem ap_buffer, const size_t ap_offset,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDspr2(layout, triangle,
                     n,
                     alpha,
                     x_buffer, x_offset, static_cast<int>(x_inc),
                     y_buffer, y_offset, static_cast<int>(y_inc),
                     ap_buffer, ap_offset,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// Forwards the clBLAS calls for SGEMM/DGEMM/CGEMM/ZGEMM
clblasStatus clblasXgemm(const clblasOrder layout, const clblasTranspose a_transpose, const clblasTranspose b_transpose,
                         const size_t m, const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSgemm(layout, a_transpose, b_transpose,
                     m, n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgemm(const clblasOrder layout, const clblasTranspose a_transpose, const clblasTranspose b_transpose,
                         const size_t m, const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDgemm(layout, a_transpose, b_transpose,
                     m, n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgemm(const clblasOrder layout, const clblasTranspose a_transpose, const clblasTranspose b_transpose,
                         const size_t m, const size_t n, const size_t k,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCgemm(layout, a_transpose, b_transpose,
                     m, n, k,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     cl_float2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXgemm(const clblasOrder layout, const clblasTranspose a_transpose, const clblasTranspose b_transpose,
                         const size_t m, const size_t n, const size_t k,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZgemm(layout, a_transpose, b_transpose,
                     m, n, k,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     cl_double2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSYMM/DSYMM/CSYMM/ZSYMM
clblasStatus clblasXsymm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle,
                         const size_t m, const size_t n,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsymm(layout, side, triangle,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsymm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle,
                         const size_t m, const size_t n,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsymm(layout, side, triangle,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsymm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCsymm(layout, side, triangle,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     cl_float2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsymm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZsymm(layout, side, triangle,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     cl_double2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHEMM/ZHEMM
clblasStatus clblasXhemm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasChemm(layout, side, triangle,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     cl_float2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXhemm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZhemm(layout, side, triangle,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     cl_double2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSYRK/DSYRK/CSYRK/ZSYRK
clblasStatus clblasXsyrk(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose,
                         const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsyrk(layout, triangle, a_transpose,
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyrk(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsyrk(layout, triangle, a_transpose,
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyrk(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose,
                         const size_t n, const size_t k,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCsyrk(layout, triangle, a_transpose,
                     n, k,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     cl_float2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyrk(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose,
                         const size_t n, const size_t k,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZsyrk(layout, triangle, a_transpose,
                     n, k,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     cl_double2{{beta.real(), beta.imag()}},
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHERK/ZHERK
clblasStatus clblasXherk(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose,
                         const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCherk(layout, triangle, a_transpose,
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXherk(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose a_transpose,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZherk(layout, triangle, a_transpose,
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for SSYR2K/DSYR2K/CSYR2K/ZSYR2K
clblasStatus clblasXsyr2k(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose ab_transpose,
                          const size_t n, const size_t k,
                          const float alpha,
                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                          const float beta,
                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                          cl_uint num_queues, cl_command_queue *queues,
                          cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasSsyr2k(layout, triangle, ab_transpose,
                      n, k,
                      alpha,
                      a_buffer, a_offset, a_ld,
                      b_buffer, b_offset, b_ld,
                      beta,
                      c_buffer, c_offset, c_ld,
                      num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyr2k(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose ab_transpose,
                          const size_t n, const size_t k,
                          const double alpha,
                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                          const double beta,
                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                          cl_uint num_queues, cl_command_queue *queues,
                          cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDsyr2k(layout, triangle, ab_transpose,
                      n, k,
                      alpha,
                      a_buffer, a_offset, a_ld,
                      b_buffer, b_offset, b_ld,
                      beta,
                      c_buffer, c_offset, c_ld,
                      num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyr2k(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose ab_transpose,
                          const size_t n, const size_t k,
                          const float2 alpha,
                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                          const float2 beta,
                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                          cl_uint num_queues, cl_command_queue *queues,
                          cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCsyr2k(layout, triangle, ab_transpose,
                      n, k,
                      cl_float2{{alpha.real(), alpha.imag()}},
                      a_buffer, a_offset, a_ld,
                      b_buffer, b_offset, b_ld,
                      cl_float2{{beta.real(), beta.imag()}},
                      c_buffer, c_offset, c_ld,
                      num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXsyr2k(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose ab_transpose,
                          const size_t n, const size_t k,
                          const double2 alpha,
                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                          const double2 beta,
                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                          cl_uint num_queues, cl_command_queue *queues,
                          cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZsyr2k(layout, triangle, ab_transpose,
                      n, k,
                      cl_double2{{alpha.real(), alpha.imag()}},
                      a_buffer, a_offset, a_ld,
                      b_buffer, b_offset, b_ld,
                      cl_double2{{beta.real(), beta.imag()}},
                      c_buffer, c_offset, c_ld,
                      num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for CHER2K/ZHER2K
clblasStatus clblasXher2k(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose ab_transpose,
                          const size_t n, const size_t k,
                          const float2 alpha,
                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                          const float beta,
                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                          cl_uint num_queues, cl_command_queue *queues,
                          cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCher2k(layout, triangle, ab_transpose,
                      n, k,
                      cl_float2{{alpha.real(), alpha.imag()}},
                      a_buffer, a_offset, a_ld,
                      b_buffer, b_offset, b_ld,
                      beta,
                      c_buffer, c_offset, c_ld,
                      num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXher2k(const clblasOrder layout, const clblasUplo triangle, const clblasTranspose ab_transpose,
                          const size_t n, const size_t k,
                          const double2 alpha,
                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                          const double beta,
                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                          cl_uint num_queues, cl_command_queue *queues,
                          cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZher2k(layout, triangle, ab_transpose,
                      n, k,
                      cl_double2{{alpha.real(), alpha.imag()}},
                      a_buffer, a_offset, a_ld,
                      b_buffer, b_offset, b_ld,
                      beta,
                      c_buffer, c_offset, c_ld,
                      num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STRMM/DTRMM/CTRMM/ZTRMM
clblasStatus clblasXtrmm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasStrmm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXtrmm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDtrmm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXtrmm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCtrmm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXtrmm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZtrmm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// Forwards the clBLAS calls for STRSM/DTRSM/CTRSM/ZTRSM
clblasStatus clblasXtrsm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasStrsm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXtrsm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasDtrsm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXtrsm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasCtrsm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     cl_float2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}
clblasStatus clblasXtrsm(const clblasOrder layout, const clblasSide side, const clblasUplo triangle, const clblasTranspose a_transpose, const clblasDiag diagonal,
                         const size_t m, const size_t n,
                         const double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         cl_uint num_queues, cl_command_queue *queues,
                         cl_uint num_wait_events, const cl_event *wait_events, cl_event *events) {
  return clblasZtrsm(layout, side, triangle, a_transpose, diagonal,
                     m, n,
                     cl_double2{{alpha.real(), alpha.imag()}},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     num_queues, queues, num_wait_events, wait_events, events);
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_WRAPPER_CLBLAS_H_
#endif
