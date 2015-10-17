
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the plain C BLAS API calls. This forwards the calls to the C++ API.
//
// =================================================================================================

#include <string>

extern "C" {
  #include "clblast_c.h"
}
#include "clblast.h"
#include "internal/utilities.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// SWAP
StatusCode CLBlastSswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<float>(n,
                                     x_buffer, x_offset, x_inc,
                                     y_buffer, y_offset, y_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<double>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<float2>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<double2>(n,
                                       x_buffer, x_offset, x_inc,
                                       y_buffer, y_offset, y_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// SCAL
StatusCode CLBlastSscal(const size_t n,
                        const float alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDscal(const size_t n,
                        const double alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCscal(const size_t n,
                        const cl_float2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZscal(const size_t n,
                        const cl_double2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              double2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// COPY
StatusCode CLBlastScopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<float>(n,
                                     x_buffer, x_offset, x_inc,
                                     y_buffer, y_offset, y_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<double>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<float2>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<double2>(n,
                                       x_buffer, x_offset, x_inc,
                                       y_buffer, y_offset, y_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// AXPY
StatusCode CLBlastSaxpy(const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Axpy(n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDaxpy(const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Axpy(n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCaxpy(const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Axpy(n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZaxpy(const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Axpy(n,
                              double2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// DOT
StatusCode CLBlastSdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Dot<float>(n,
                                    dot_buffer, dot_offset,
                                    x_buffer, x_offset, x_inc,
                                    y_buffer, y_offset, y_inc,
                                    queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Dot<double>(n,
                                     dot_buffer, dot_offset,
                                     x_buffer, x_offset, x_inc,
                                     y_buffer, y_offset, y_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}

// DOTU
StatusCode CLBlastCdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Dotu<float2>(n,
                                      dot_buffer, dot_offset,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Dotu<double2>(n,
                                       dot_buffer, dot_offset,
                                       x_buffer, x_offset, x_inc,
                                       y_buffer, y_offset, y_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// DOTC
StatusCode CLBlastCdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Dotc<float2>(n,
                                      dot_buffer, dot_offset,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Dotc<double2>(n,
                                       dot_buffer, dot_offset,
                                       x_buffer, x_offset, x_inc,
                                       y_buffer, y_offset, y_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// GEMV
StatusCode CLBlastSgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              float2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              double2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// GBMV
StatusCode CLBlastSgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n, kl, ku,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n, kl, ku,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n, kl, ku,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              float2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              m, n, kl, ku,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              double2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HEMV
StatusCode CLBlastChemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hemv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              float2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZhemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hemv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              double2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HBMV
StatusCode CLBlastChbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n, k,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              float2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZhbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n, k,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              double2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HPMV
StatusCode CLBlastChpmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hpmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              float2{alpha.s[0], alpha.s[1]},
                              ap_buffer, ap_offset,
                              x_buffer, x_offset, x_inc,
                              float2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZhpmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hpmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              double2{alpha.s[0], alpha.s[1]},
                              ap_buffer, ap_offset,
                              x_buffer, x_offset, x_inc,
                              double2{beta.s[0], beta.s[1]},
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SYMV
StatusCode CLBlastSsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Symv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Symv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SBMV
StatusCode CLBlastSsbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Sbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Sbmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SPMV
StatusCode CLBlastSspmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const float beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Spmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              ap_buffer, ap_offset,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDspmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Spmv(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              ap_buffer, ap_offset,
                              x_buffer, x_offset, x_inc,
                              beta,
                              y_buffer, y_offset, y_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// TRMV
StatusCode CLBlastStrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmv<float>(static_cast<clblast::Layout>(layout),
                                     static_cast<clblast::Triangle>(triangle),
                                     static_cast<clblast::Transpose>(a_transpose),
                                     static_cast<clblast::Diagonal>(diagonal),
                                     n,
                                     a_buffer, a_offset, a_ld,
                                     x_buffer, x_offset, x_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmv<double>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmv<float2>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmv<double2>(static_cast<clblast::Layout>(layout),
                                       static_cast<clblast::Triangle>(triangle),
                                       static_cast<clblast::Transpose>(a_transpose),
                                       static_cast<clblast::Diagonal>(diagonal),
                                       n,
                                       a_buffer, a_offset, a_ld,
                                       x_buffer, x_offset, x_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// TBMV
StatusCode CLBlastStbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbmv<float>(static_cast<clblast::Layout>(layout),
                                     static_cast<clblast::Triangle>(triangle),
                                     static_cast<clblast::Transpose>(a_transpose),
                                     static_cast<clblast::Diagonal>(diagonal),
                                     n, k,
                                     a_buffer, a_offset, a_ld,
                                     x_buffer, x_offset, x_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbmv<double>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n, k,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbmv<float2>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n, k,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbmv<double2>(static_cast<clblast::Layout>(layout),
                                       static_cast<clblast::Triangle>(triangle),
                                       static_cast<clblast::Transpose>(a_transpose),
                                       static_cast<clblast::Diagonal>(diagonal),
                                       n, k,
                                       a_buffer, a_offset, a_ld,
                                       x_buffer, x_offset, x_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// TPMV
StatusCode CLBlastStpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpmv<float>(static_cast<clblast::Layout>(layout),
                                     static_cast<clblast::Triangle>(triangle),
                                     static_cast<clblast::Transpose>(a_transpose),
                                     static_cast<clblast::Diagonal>(diagonal),
                                     n,
                                     ap_buffer, ap_offset,
                                     x_buffer, x_offset, x_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpmv<double>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      ap_buffer, ap_offset,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpmv<float2>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      ap_buffer, ap_offset,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpmv<double2>(static_cast<clblast::Layout>(layout),
                                       static_cast<clblast::Triangle>(triangle),
                                       static_cast<clblast::Transpose>(a_transpose),
                                       static_cast<clblast::Diagonal>(diagonal),
                                       n,
                                       ap_buffer, ap_offset,
                                       x_buffer, x_offset, x_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// TRSV
StatusCode CLBlastStrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsv<float>(static_cast<clblast::Layout>(layout),
                                     static_cast<clblast::Triangle>(triangle),
                                     static_cast<clblast::Transpose>(a_transpose),
                                     static_cast<clblast::Diagonal>(diagonal),
                                     n,
                                     a_buffer, a_offset, a_ld,
                                     x_buffer, x_offset, x_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsv<double>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsv<float2>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsv<double2>(static_cast<clblast::Layout>(layout),
                                       static_cast<clblast::Triangle>(triangle),
                                       static_cast<clblast::Transpose>(a_transpose),
                                       static_cast<clblast::Diagonal>(diagonal),
                                       n,
                                       a_buffer, a_offset, a_ld,
                                       x_buffer, x_offset, x_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// TBSV
StatusCode CLBlastStbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbsv<float>(static_cast<clblast::Layout>(layout),
                                     static_cast<clblast::Triangle>(triangle),
                                     static_cast<clblast::Transpose>(a_transpose),
                                     static_cast<clblast::Diagonal>(diagonal),
                                     n, k,
                                     a_buffer, a_offset, a_ld,
                                     x_buffer, x_offset, x_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbsv<double>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n, k,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbsv<float2>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n, k,
                                      a_buffer, a_offset, a_ld,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tbsv<double2>(static_cast<clblast::Layout>(layout),
                                       static_cast<clblast::Triangle>(triangle),
                                       static_cast<clblast::Transpose>(a_transpose),
                                       static_cast<clblast::Diagonal>(diagonal),
                                       n, k,
                                       a_buffer, a_offset, a_ld,
                                       x_buffer, x_offset, x_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// TPSV
StatusCode CLBlastStpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpsv<float>(static_cast<clblast::Layout>(layout),
                                     static_cast<clblast::Triangle>(triangle),
                                     static_cast<clblast::Transpose>(a_transpose),
                                     static_cast<clblast::Diagonal>(diagonal),
                                     n,
                                     ap_buffer, ap_offset,
                                     x_buffer, x_offset, x_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpsv<double>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      ap_buffer, ap_offset,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpsv<float2>(static_cast<clblast::Layout>(layout),
                                      static_cast<clblast::Triangle>(triangle),
                                      static_cast<clblast::Transpose>(a_transpose),
                                      static_cast<clblast::Diagonal>(diagonal),
                                      n,
                                      ap_buffer, ap_offset,
                                      x_buffer, x_offset, x_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Tpsv<double2>(static_cast<clblast::Layout>(layout),
                                       static_cast<clblast::Triangle>(triangle),
                                       static_cast<clblast::Transpose>(a_transpose),
                                       static_cast<clblast::Diagonal>(diagonal),
                                       n,
                                       ap_buffer, ap_offset,
                                       x_buffer, x_offset, x_inc,
                                       queue, event);
  return static_cast<StatusCode>(status);
}

// GER
StatusCode CLBlastSger(const Layout layout,
                       const size_t m, const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Ger(static_cast<clblast::Layout>(layout),
                             m, n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             y_buffer, y_offset, y_inc,
                             a_buffer, a_offset, a_ld,
                             queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDger(const Layout layout,
                       const size_t m, const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Ger(static_cast<clblast::Layout>(layout),
                             m, n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             y_buffer, y_offset, y_inc,
                             a_buffer, a_offset, a_ld,
                             queue, event);
  return static_cast<StatusCode>(status);
}

// GERU
StatusCode CLBlastCgeru(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Geru(static_cast<clblast::Layout>(layout),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZgeru(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Geru(static_cast<clblast::Layout>(layout),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// GERC
StatusCode CLBlastCgerc(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gerc(static_cast<clblast::Layout>(layout),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZgerc(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gerc(static_cast<clblast::Layout>(layout),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HER
StatusCode CLBlastCher(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Her(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             a_buffer, a_offset, a_ld,
                             queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZher(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Her(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             a_buffer, a_offset, a_ld,
                             queue, event);
  return static_cast<StatusCode>(status);
}

// HPR
StatusCode CLBlastChpr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hpr(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             ap_buffer, ap_offset,
                             queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZhpr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hpr(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             ap_buffer, ap_offset,
                             queue, event);
  return static_cast<StatusCode>(status);
}

// HER2
StatusCode CLBlastCher2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Her2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZher2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Her2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              double2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HPR2
StatusCode CLBlastChpr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hpr2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              ap_buffer, ap_offset,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZhpr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hpr2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              double2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              ap_buffer, ap_offset,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SYR
StatusCode CLBlastSsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             a_buffer, a_offset, a_ld,
                             queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             a_buffer, a_offset, a_ld,
                             queue, event);
  return static_cast<StatusCode>(status);
}

// SPR
StatusCode CLBlastSspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Spr(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             ap_buffer, ap_offset,
                             queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Spr(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             n,
                             alpha,
                             x_buffer, x_offset, x_inc,
                             ap_buffer, ap_offset,
                             queue, event);
  return static_cast<StatusCode>(status);
}

// SYR2
StatusCode CLBlastSsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              a_buffer, a_offset, a_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SPR2
StatusCode CLBlastSspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Spr2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              ap_buffer, ap_offset,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Spr2(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              y_buffer, y_offset, y_inc,
                              ap_buffer, ap_offset,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// GEMM
StatusCode CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              float2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              double2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SYMM
StatusCode CLBlastSsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Symm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Symm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Symm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              float2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Symm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              double2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HEMM
StatusCode CLBlastChemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              float2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZhemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Hemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              double2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SYRK
StatusCode CLBlastSsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syrk(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syrk(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syrk(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              n, k,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              float2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syrk(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              n, k,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              double2{beta.s[0], beta.s[1]},
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// HERK
StatusCode CLBlastCherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Herk(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Herk(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              n, k,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              beta,
                              c_buffer, c_offset, c_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// SYR2K
StatusCode CLBlastSsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const float alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                               static_cast<clblast::Triangle>(triangle),
                               static_cast<clblast::Transpose>(ab_transpose),
                               n, k,
                               alpha,
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               beta,
                               c_buffer, c_offset, c_ld,
                               queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                               static_cast<clblast::Triangle>(triangle),
                               static_cast<clblast::Transpose>(ab_transpose),
                               n, k,
                               alpha,
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               beta,
                               c_buffer, c_offset, c_ld,
                               queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                               static_cast<clblast::Triangle>(triangle),
                               static_cast<clblast::Transpose>(ab_transpose),
                               n, k,
                               float2{alpha.s[0], alpha.s[1]},
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               float2{beta.s[0], beta.s[1]},
                               c_buffer, c_offset, c_ld,
                               queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Syr2k(static_cast<clblast::Layout>(layout),
                               static_cast<clblast::Triangle>(triangle),
                               static_cast<clblast::Transpose>(ab_transpose),
                               n, k,
                               double2{alpha.s[0], alpha.s[1]},
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               double2{beta.s[0], beta.s[1]},
                               c_buffer, c_offset, c_ld,
                               queue, event);
  return static_cast<StatusCode>(status);
}

// HER2K
StatusCode CLBlastCher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const float beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Her2k(static_cast<clblast::Layout>(layout),
                               static_cast<clblast::Triangle>(triangle),
                               static_cast<clblast::Transpose>(ab_transpose),
                               n, k,
                               float2{alpha.s[0], alpha.s[1]},
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               beta,
                               c_buffer, c_offset, c_ld,
                               queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Her2k(static_cast<clblast::Layout>(layout),
                               static_cast<clblast::Triangle>(triangle),
                               static_cast<clblast::Transpose>(ab_transpose),
                               n, k,
                               double2{alpha.s[0], alpha.s[1]},
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               beta,
                               c_buffer, c_offset, c_ld,
                               queue, event);
  return static_cast<StatusCode>(status);
}

// TRMM
StatusCode CLBlastStrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trmm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// TRSM
StatusCode CLBlastStrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              alpha,
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Trsm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Side>(side),
                              static_cast<clblast::Triangle>(triangle),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Diagonal>(diagonal),
                              m, n,
                              double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// =================================================================================================
