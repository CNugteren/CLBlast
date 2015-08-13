
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

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// GEMM
StatusCode CLBlastSgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k, const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k, alpha,
                              a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta,
                              c_buffer, c_offset, c_ld, queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k, const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k, alpha,
                              a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta,
                              c_buffer, c_offset, c_ld, queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k, const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto cl_alpha = clblast::float2{alpha.s[0], alpha.s[1]};
  auto cl_beta = clblast::float2{beta.s[0], beta.s[1]};
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k, cl_alpha,
                              a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, cl_beta,
                              c_buffer, c_offset, c_ld, queue, event);
  return static_cast<StatusCode>(status);
}
StatusCode CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k, const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  auto cl_alpha = clblast::double2{alpha.s[0], alpha.s[1]};
  auto cl_beta = clblast::double2{beta.s[0], beta.s[1]};
  auto status = clblast::Gemm(static_cast<clblast::Layout>(layout),
                              static_cast<clblast::Transpose>(a_transpose),
                              static_cast<clblast::Transpose>(b_transpose),
                              m, n, k, cl_alpha,
                              a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, cl_beta,
                              c_buffer, c_offset, c_ld, queue, event);
  return static_cast<StatusCode>(status);
}

// =================================================================================================
