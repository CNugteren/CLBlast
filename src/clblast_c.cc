
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

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

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
                              clblast::float2{alpha.s[0], alpha.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              clblast::float2{beta.s[0], beta.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              clblast::double2{beta.s[0], beta.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              clblast::float2{beta.s[0], beta.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              x_buffer, x_offset, x_inc,
                              clblast::double2{beta.s[0], beta.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              clblast::float2{beta.s[0], beta.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              clblast::double2{beta.s[0], beta.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              clblast::float2{beta.s[0], beta.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              clblast::double2{beta.s[0], beta.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              clblast::float2{beta.s[0], beta.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              clblast::double2{beta.s[0], beta.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              clblast::float2{beta.s[0], beta.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              clblast::double2{beta.s[0], beta.s[1]},
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
                               clblast::float2{alpha.s[0], alpha.s[1]},
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               clblast::float2{beta.s[0], beta.s[1]},
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
                               clblast::double2{alpha.s[0], alpha.s[1]},
                               a_buffer, a_offset, a_ld,
                               b_buffer, b_offset, b_ld,
                               clblast::double2{beta.s[0], beta.s[1]},
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
                               clblast::float2{alpha.s[0], alpha.s[1]},
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
                               clblast::double2{alpha.s[0], alpha.s[1]},
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
                              clblast::float2{alpha.s[0], alpha.s[1]},
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
                              clblast::double2{alpha.s[0], alpha.s[1]},
                              a_buffer, a_offset, a_ld,
                              b_buffer, b_offset, b_ld,
                              queue, event);
  return static_cast<StatusCode>(status);
}

// =================================================================================================
