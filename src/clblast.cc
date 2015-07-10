
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
#include "internal/routines/xsyrk.h"
#include "internal/routines/xherk.h"
#include "internal/routines/xsyr2k.h"
#include "internal/routines/xher2k.h"
#include "internal/routines/xtrmm.h"

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
StatusCode Gemv(const Layout layout, const Transpose a_transpose,
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
  return routine.DoGemv(layout, a_transpose, m, n, alpha,
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
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
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
  return routine.DoGemm(layout, a_transpose, b_transpose, m, n, k, alpha,
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

// =================================================================================================

// SYRK
template <typename T>
StatusCode Syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyrk<T>(queue_cpp, event_cpp);

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
  return routine.DoSyrk(layout, triangle, a_transpose, n, k, alpha,
                        Buffer(a_buffer), a_offset, a_ld, beta,
                        Buffer(c_buffer), c_offset, c_ld);
}
template StatusCode Syrk<float>(const Layout, const Triangle, const Transpose,
                                const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t, const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Syrk<double>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t, const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Syrk<float2>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t, const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Syrk<double2>(const Layout, const Triangle, const Transpose,
                                  const size_t, const size_t, const double2,
                                  const cl_mem, const size_t, const size_t, const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================

// HERK
template <typename T>
StatusCode Herk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xherk<std::complex<T>,T>(queue_cpp, event_cpp);

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
  return routine.DoHerk(layout, triangle, a_transpose, n, k, alpha,
                        Buffer(a_buffer), a_offset, a_ld, beta,
                        Buffer(c_buffer), c_offset, c_ld);
}
template StatusCode Herk<float>(const Layout, const Triangle, const Transpose,
                                const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t, const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Herk<double>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t, const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// =================================================================================================

// SYR2K
template <typename T>
StatusCode Syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k, const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyr2k<T>(queue_cpp, event_cpp);

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
  return routine.DoSyr2k(layout, triangle, ab_transpose, n, k, alpha,
                         Buffer(a_buffer), a_offset, a_ld,
                         Buffer(b_buffer), b_offset, b_ld, beta,
                         Buffer(c_buffer), c_offset, c_ld);
}
template StatusCode Syr2k<float>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t, const float,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const float,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Syr2k<double>(const Layout, const Triangle, const Transpose,
                                  const size_t, const size_t, const double,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t, const double,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);
template StatusCode Syr2k<float2>(const Layout, const Triangle, const Transpose,
                                  const size_t, const size_t, const float2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t, const float2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);
template StatusCode Syr2k<double2>(const Layout, const Triangle, const Transpose,
                                   const size_t, const size_t, const double2,
                                   const cl_mem, const size_t, const size_t,
                                   const cl_mem, const size_t, const size_t, const double2,
                                   cl_mem, const size_t, const size_t,
                                   cl_command_queue*, cl_event*);

// =================================================================================================

// SYR2K
template <typename T, typename U>
StatusCode Her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k, const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const U beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xher2k<T,U>(queue_cpp, event_cpp);

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
  return routine.DoHer2k(layout, triangle, ab_transpose, n, k, alpha,
                         Buffer(a_buffer), a_offset, a_ld,
                         Buffer(b_buffer), b_offset, b_ld, beta,
                         Buffer(c_buffer), c_offset, c_ld);
}
template StatusCode Her2k<float2,float>(const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t, const float2,
                                        const cl_mem, const size_t, const size_t,
                                        const cl_mem, const size_t, const size_t, const float,
                                        cl_mem, const size_t, const size_t,
                                        cl_command_queue*, cl_event*);
template StatusCode Her2k<double2,double>(const Layout, const Triangle, const Transpose,
                                          const size_t, const size_t, const double2,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t, const double,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// =================================================================================================

// TRMM
template <typename T>
StatusCode Trmm(const Layout layout, const Side side, const Triangle triangle,
                const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtrmm<T>(queue_cpp, event_cpp);

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
  return routine.DoTrmm(layout, side, triangle, a_transpose, diagonal, m, n, alpha,
                        Buffer(a_buffer), a_offset, a_ld,
                        Buffer(b_buffer), b_offset, b_ld);
}
template StatusCode Trmm<float>(const Layout, const Side, const Triangle,
                                const Transpose, const Diagonal,
                                const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Trmm<double>(const Layout, const Side, const Triangle,
                                 const Transpose, const Diagonal,
                                 const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trmm<float2>(const Layout, const Side, const Triangle,
                                 const Transpose, const Diagonal,
                                 const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trmm<double2>(const Layout, const Side, const Triangle,
                                  const Transpose, const Diagonal,
                                  const size_t, const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================

// TRSM
template <typename T>
StatusCode Trsm(const Layout layout, const Side side, const Triangle triangle,
                const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = CommandQueue(*queue);
  auto event_cpp = Event(*event);
  /*
  auto routine = Xtrsm<T>(queue_cpp, event_cpp);

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
  return routine.DoTrsm(layout, side, triangle, a_transpose, diagonal, m, n, alpha,
                        Buffer(a_buffer), a_offset, a_ld,
                        Buffer(b_buffer), b_offset, b_ld);
  */
  return StatusCode::kNotImplemented;
}
template StatusCode Trsm<float>(const Layout, const Side, const Triangle,
                                const Transpose, const Diagonal,
                                const size_t, const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Trsm<double>(const Layout, const Side, const Triangle,
                                 const Transpose, const Diagonal,
                                 const size_t, const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trsm<float2>(const Layout, const Side, const Triangle,
                                 const Transpose, const Diagonal,
                                 const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trsm<double2>(const Layout, const Side, const Triangle,
                                  const Transpose, const Diagonal,
                                  const size_t, const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================
} // namespace clblast
