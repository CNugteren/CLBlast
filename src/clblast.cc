
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
#include "internal/routines/level1/xswap.h"
#include "internal/routines/level1/xscal.h"
#include "internal/routines/level1/xcopy.h"
#include "internal/routines/level1/xaxpy.h"

// BLAS level-2 includes
#include "internal/routines/level2/xgemv.h"
#include "internal/routines/level2/xhemv.h"
#include "internal/routines/level2/xsymv.h"

// BLAS level-3 includes
#include "internal/routines/level3/xgemm.h"
#include "internal/routines/level3/xsymm.h"
#include "internal/routines/level3/xhemm.h"
#include "internal/routines/level3/xsyrk.h"
#include "internal/routines/level3/xherk.h"
#include "internal/routines/level3/xsyr2k.h"
#include "internal/routines/level3/xher2k.h"
#include "internal/routines/level3/xtrmm.h"

namespace clblast {
// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// SWAP
template <typename T>
StatusCode Swap(const size_t n,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xswap<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSwap(n,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Swap<float>(const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue* queue, cl_event* event);
template StatusCode Swap<double>(const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Swap<float2>(const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Swap<double2>(const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue* queue, cl_event* event);

// SCAL
template <typename T>
StatusCode Scal(const size_t n,
                const T alpha,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xscal<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoScal(n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode Scal<float>(const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue* queue, cl_event* event);
template StatusCode Scal<double>(const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Scal<float2>(const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Scal<double2>(const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue* queue, cl_event* event);

// COPY
template <typename T>
StatusCode Copy(const size_t n,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xcopy<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoCopy(n,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Copy<float>(const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue* queue, cl_event* event);
template StatusCode Copy<double>(const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Copy<float2>(const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Copy<double2>(const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue* queue, cl_event* event);

// AXPY
template <typename T>
StatusCode Axpy(const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xaxpy<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoAxpy(n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Axpy<float>(const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue* queue, cl_event* event);
template StatusCode Axpy<double>(const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Axpy<float2>(const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue* queue, cl_event* event);
template StatusCode Axpy<double2>(const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// GEMV
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgemv<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoGemv(layout, a_transpose, m, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc, beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
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

// HEMV
template <typename T>
StatusCode Hemv(const Layout layout, const Triangle triangle,
                const size_t n, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {

  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhemv<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoHemv(layout, triangle, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc, beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Hemv<float2>(const Layout, const Triangle,
                                 const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hemv<double2>(const Layout, const Triangle,
                                  const size_t, const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t, const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================

// SYMV
template <typename T>
StatusCode Symv(const Layout layout, const Triangle triangle,
                const size_t n, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {

  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsymv<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoSymv(layout, triangle, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc, beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Symv<float>(const Layout, const Triangle,
                                const size_t, const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t, const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Symv<double>(const Layout, const Triangle,
                                 const size_t, const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// GEMM
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                const size_t m, const size_t n, const size_t k, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgemm<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoGemm(layout, a_transpose, b_transpose, m, n, k, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld, beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
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
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsymm<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoSymm(layout, side, triangle, m, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld, beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
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

// HEMM
template <typename T>
StatusCode Hemm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n, const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhemm<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoHemm(layout, side, triangle, m, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld, beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Hemm<float2>(const Layout, const Side, const Triangle,
                                 const size_t, const size_t, const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t, const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hemm<double2>(const Layout, const Side, const Triangle,
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
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyrk<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoSyrk(layout, triangle, a_transpose, n, k, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld, beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
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
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xherk<std::complex<T>,T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoHerk(layout, triangle, a_transpose, n, k, alpha,
                        Buffer<std::complex<T>>(a_buffer), a_offset, a_ld, beta,
                        Buffer<std::complex<T>>(c_buffer), c_offset, c_ld);
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
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyr2k<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoSyr2k(layout, triangle, ab_transpose, n, k, alpha,
                         Buffer<T>(a_buffer), a_offset, a_ld,
                         Buffer<T>(b_buffer), b_offset, b_ld, beta,
                         Buffer<T>(c_buffer), c_offset, c_ld);
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
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xher2k<T,U>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoHer2k(layout, triangle, ab_transpose, n, k, alpha,
                         Buffer<T>(a_buffer), a_offset, a_ld,
                         Buffer<T>(b_buffer), b_offset, b_ld, beta,
                         Buffer<T>(c_buffer), c_offset, c_ld);
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
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtrmm<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoTrmm(layout, side, triangle, a_transpose, diagonal, m, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld);
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
/*
template <typename T>
StatusCode Trsm(const Layout layout, const Side side, const Triangle triangle,
                const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtrsm<T>(queue_cpp, event_cpp);

  // Compiles the routine's device kernels
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }

  // Runs the routine
  return routine.DoTrsm(layout, side, triangle, a_transpose, diagonal, m, n, alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld);
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
*/
// =================================================================================================
} // namespace clblast
