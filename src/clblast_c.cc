
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
#include "internal/public_api.h"
#include "internal/utilities.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// SWAP
StatusCode PUBLIC_API CLBlastSswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<float>(n,
                                     x_buffer, x_offset, x_inc,
                                     y_buffer, y_offset, y_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastDswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<double>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastCswap(const size_t n,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Swap<float2>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastZswap(const size_t n,
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
StatusCode PUBLIC_API CLBlastSscal(const size_t n,
                                   const float alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastDscal(const size_t n,
                                   const double alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              alpha,
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastCscal(const size_t n,
                                   const cl_float2 alpha,
                                   cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Scal(n,
                              float2{alpha.s[0], alpha.s[1]},
                              x_buffer, x_offset, x_inc,
                              queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastZscal(const size_t n,
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
StatusCode PUBLIC_API CLBlastScopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<float>(n,
                                     x_buffer, x_offset, x_inc,
                                     y_buffer, y_offset, y_inc,
                                     queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastDcopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<double>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastCcopy(const size_t n,
                                   const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                   cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                   cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Copy<float2>(n,
                                      x_buffer, x_offset, x_inc,
                                      y_buffer, y_offset, y_inc,
                                      queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode PUBLIC_API CLBlastZcopy(const size_t n,
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
StatusCode PUBLIC_API CLBlastSaxpy(const size_t n,
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
StatusCode PUBLIC_API CLBlastDaxpy(const size_t n,
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
StatusCode PUBLIC_API CLBlastCaxpy(const size_t n,
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
StatusCode PUBLIC_API CLBlastZaxpy(const size_t n,
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
StatusCode PUBLIC_API CLBlastSdot(const size_t n,
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
StatusCode PUBLIC_API CLBlastDdot(const size_t n,
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
StatusCode PUBLIC_API CLBlastCdotu(const size_t n,
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
StatusCode PUBLIC_API CLBlastZdotu(const size_t n,
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
StatusCode PUBLIC_API CLBlastCdotc(const size_t n,
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
StatusCode PUBLIC_API CLBlastZdotc(const size_t n,
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
StatusCode PUBLIC_API CLBlastSgemv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastDgemv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastCgemv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastZgemv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastSgbmv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastDgbmv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastCgbmv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastZgbmv(const Layout layout, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastChemv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZhemv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastChbmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZhbmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastChpmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZhpmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSsymv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDsymv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSsbmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDsbmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSspmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDspmv(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastStrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastStbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastStpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastStrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastStbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastStpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastSger(const Layout layout,
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
StatusCode PUBLIC_API CLBlastDger(const Layout layout,
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
StatusCode PUBLIC_API CLBlastCgeru(const Layout layout,
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
StatusCode PUBLIC_API CLBlastZgeru(const Layout layout,
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
StatusCode PUBLIC_API CLBlastCgerc(const Layout layout,
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
StatusCode PUBLIC_API CLBlastZgerc(const Layout layout,
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
StatusCode PUBLIC_API CLBlastCher(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZher(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastChpr(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZhpr(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastCher2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZher2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastChpr2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZhpr2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSsyr(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDsyr(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSspr(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDspr(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSsyr2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDsyr2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSspr2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDspr2(const Layout layout, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
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
StatusCode PUBLIC_API CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
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
StatusCode PUBLIC_API CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
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
StatusCode PUBLIC_API CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
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
StatusCode PUBLIC_API CLBlastSsymm(const Layout layout, const Side side, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastDsymm(const Layout layout, const Side side, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastCsymm(const Layout layout, const Side side, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZsymm(const Layout layout, const Side side, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastChemm(const Layout layout, const Side side, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastZhemm(const Layout layout, const Side side, const Triangle triangle,
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
StatusCode PUBLIC_API CLBlastSsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastDsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastCsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastZsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastCherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastZherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
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
StatusCode PUBLIC_API CLBlastSsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
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
StatusCode PUBLIC_API CLBlastDsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
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
StatusCode PUBLIC_API CLBlastCsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
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
StatusCode PUBLIC_API CLBlastZsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
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
StatusCode PUBLIC_API CLBlastCher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
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
StatusCode PUBLIC_API CLBlastZher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
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
StatusCode PUBLIC_API CLBlastStrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastStrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastDtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastCtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
StatusCode PUBLIC_API CLBlastZtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
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
