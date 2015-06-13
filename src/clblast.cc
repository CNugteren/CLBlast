
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the BLAS API calls. In all cases, it does not much more than creating
// a new object of the appropriate type, and calling the main routine on that object. It forwards
// all status codes to the caller.
//
// =================================================================================================

#include <string>

#include "clblast.h"

// BLAS level-1 includes
#include "internal/routines/xaxpy.h"

// BLAS level-2 includes
#include "internal/routines/xgemv.h"

// BLAS level-3 includes
#include "internal/routines/xgemm.h"
#include "internal/routines/xsymm.h"

namespace clblast {
// =================================================================================================
// BLAS level-1 (vector-vector) routines

// AXPY
template <typename T>
StatusCode Axpy(const size_t n, const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xaxpy<T>(queue_cpp, event_cpp);

  // Loads the kernel source-code as an include (C++11 raw string literal)
  std::string kernel_source =
  #include "kernels/xaxpy.opencl"
  auto status = routine.SetUp(kernel_source);
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoAxpy(n, alpha,
                        Buffer(x_buffer), x_offset, x_inc,
                        Buffer(y_buffer), y_offset, y_inc);
}
template StatusCode Axpy<float>(const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Axpy<double>(const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Axpy<float2>(const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Axpy<double2>(const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines

// GEMV
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose transpose_a,
                const size_t m, const size_t n, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgemv<T>(queue_cpp, event_cpp);

  // Loads the kernel source-code as an include (C++11 raw string literal)
  std::string kernel_source =
  #include "kernels/xgemv.opencl"
  auto status = routine.SetUp(kernel_source);
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoGemv(layout, transpose_a, m, n, alpha,
                        Buffer(a_buffer), a_offset, a_ld,
                        Buffer(x_buffer), x_offset, x_inc, beta,
                        Buffer(y_buffer), y_offset, y_inc);
}
template StatusCode Gemv<float>(const Layout, const Transpose,
                                const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t, const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Gemv<double>(const Layout, const Transpose,
                                 const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemv<float2>(const Layout, const Transpose,
                                 const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemv<double2>(const Layout, const Transpose,
                                  const size_t, const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t, const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines

// GEMM
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose transpose_a, const Transpose transpose_b,
                const size_t m, const size_t n, const size_t k, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgemm<T>(queue_cpp, event_cpp);

  // Loads the kernel source-code as an include (C++11 raw string literal)
  std::string common_source1 =
  #include "kernels/copy.opencl"
  std::string common_source2 =
  #include "kernels/pad.opencl"
  std::string common_source3 =
  #include "kernels/transpose.opencl"
  std::string common_source4 =
  #include "kernels/padtranspose.opencl"
  std::string kernel_source =
  #include "kernels/xgemm.opencl"
  auto status = routine.SetUp(common_source1 + common_source2 + common_source3 + common_source4 +
                              kernel_source);
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoGemm(layout, transpose_a, transpose_b, m, n, k, alpha,
                        Buffer(a_buffer), a_offset, a_ld,
                        Buffer(b_buffer), b_offset, b_ld, beta,
                        Buffer(c_buffer), c_offset, c_ld);
}
template StatusCode Gemm<float>(const Layout, const Transpose, const Transpose,
                                const size_t, const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t, const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Gemm<double>(const Layout, const Transpose, const Transpose,
                                 const size_t, const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
/*
template StatusCode Gemm<float2>(const Layout, const Transpose, const Transpose,
                                 const size_t, const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemm<double2>(const Layout, const Transpose, const Transpose,
                                  const size_t, const size_t, const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t, const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);
*/

// =================================================================================================

// SYMM
template <typename T>
StatusCode Symm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsymm<T>(queue_cpp, event_cpp);

  // Loads the kernel source-code as an include (C++11 raw string literal)
  std::string common_source1 =
  #include "kernels/copy.opencl"
  std::string common_source2 =
  #include "kernels/pad.opencl"
  std::string common_source3 =
  #include "kernels/transpose.opencl"
  std::string common_source4 =
  #include "kernels/padtranspose.opencl"
  std::string kernel_source =
  #include "kernels/xgemm.opencl"
  auto status = routine.SetUp(common_source1 + common_source2 + common_source3 + common_source4 +
                            kernel_source);
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoSymm(layout, side, triangle, m, n, alpha,
                        Buffer(a_buffer), a_offset, a_ld,
                        Buffer(b_buffer), b_offset, b_ld, beta,
                        Buffer(c_buffer), c_offset, c_ld);
}
template StatusCode Symm<float>(const Layout, const Side, const Triangle,
                                const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t, const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Symm<double>(const Layout, const Side, const Triangle,
                                 const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
/*
template StatusCode Symm<float2>(const Layout, const Side, const Triangle,
                                 const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Symm<double2>(const Layout, const Side, const Triangle,
                                  const size_t, const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t, const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);
*/

// =================================================================================================
} // namespace clblast
