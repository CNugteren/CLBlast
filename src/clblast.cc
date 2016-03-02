
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
#include "internal/routines/level1/xdot.h"
#include "internal/routines/level1/xdotu.h"
#include "internal/routines/level1/xdotc.h"

// BLAS level-2 includes
#include "internal/routines/level2/xgemv.h"
#include "internal/routines/level2/xgbmv.h"
#include "internal/routines/level2/xhemv.h"
#include "internal/routines/level2/xhbmv.h"
#include "internal/routines/level2/xhpmv.h"
#include "internal/routines/level2/xsymv.h"
#include "internal/routines/level2/xsbmv.h"
#include "internal/routines/level2/xspmv.h"
#include "internal/routines/level2/xtrmv.h"
#include "internal/routines/level2/xtbmv.h"
#include "internal/routines/level2/xtpmv.h"
#include "internal/routines/level2/xger.h"
#include "internal/routines/level2/xgeru.h"
#include "internal/routines/level2/xgerc.h"
#include "internal/routines/level2/xher.h"
#include "internal/routines/level2/xhpr.h"
#include "internal/routines/level2/xher2.h"
#include "internal/routines/level2/xsyr.h"
#include "internal/routines/level2/xspr.h"
#include "internal/routines/level2/xsyr2.h"

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

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP
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
                                cl_command_queue*, cl_event*);
template StatusCode Swap<double>(const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Swap<float2>(const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Swap<double2>(const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL
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
                                cl_command_queue*, cl_event*);
template StatusCode Scal<double>(const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Scal<float2>(const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Scal<double2>(const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY
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
                                cl_command_queue*, cl_event*);
template StatusCode Copy<double>(const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Copy<float2>(const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Copy<double2>(const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY
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
                                cl_command_queue*, cl_event*);
template StatusCode Axpy<double>(const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Axpy<float2>(const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Axpy<double2>(const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Dot product of two vectors: SDOT/DDOT
template <typename T>
StatusCode Dot(const size_t n,
               cl_mem dot_buffer, const size_t dot_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xdot<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoDot(n,
                       Buffer<T>(dot_buffer), dot_offset,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Dot<float>(const size_t,
                               cl_mem, const size_t,
                               const cl_mem, const size_t, const size_t,
                               const cl_mem, const size_t, const size_t,
                               cl_command_queue*, cl_event*);
template StatusCode Dot<double>(const size_t,
                                cl_mem, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);

// Dot product of two complex vectors: CDOTU/ZDOTU
template <typename T>
StatusCode Dotu(const size_t n,
                cl_mem dot_buffer, const size_t dot_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xdotu<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoDotu(n,
                        Buffer<T>(dot_buffer), dot_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Dotu<float2>(const size_t,
                                 cl_mem, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Dotu<double2>(const size_t,
                                  cl_mem, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
template <typename T>
StatusCode Dotc(const size_t n,
                cl_mem dot_buffer, const size_t dot_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xdotc<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoDotc(n,
                        Buffer<T>(dot_buffer), dot_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Dotc<float2>(const size_t,
                                 cl_mem, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Dotc<double2>(const size_t,
                                  cl_mem, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgemv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGemv(layout, a_transpose,
                        m, n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Gemv<float>(const Layout, const Transpose,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Gemv<double>(const Layout, const Transpose,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemv<float2>(const Layout, const Transpose,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemv<double2>(const Layout, const Transpose,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV
template <typename T>
StatusCode Gbmv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n, const size_t kl, const size_t ku,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgbmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGbmv(layout, a_transpose,
                        m, n, kl, ku,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Gbmv<float>(const Layout, const Transpose,
                                const size_t, const size_t, const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Gbmv<double>(const Layout, const Transpose,
                                 const size_t, const size_t, const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gbmv<float2>(const Layout, const Transpose,
                                 const size_t, const size_t, const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gbmv<double2>(const Layout, const Transpose,
                                  const size_t, const size_t, const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
template <typename T>
StatusCode Hemv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhemv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHemv(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Hemv<float2>(const Layout, const Triangle,
                                 const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hemv<double2>(const Layout, const Triangle,
                                  const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
template <typename T>
StatusCode Hbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhbmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHbmv(layout, triangle,
                        n, k,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Hbmv<float2>(const Layout, const Triangle,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hbmv<double2>(const Layout, const Triangle,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
template <typename T>
StatusCode Hpmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem ap_buffer, const size_t ap_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhpmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHpmv(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(ap_buffer), ap_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Hpmv<float2>(const Layout, const Triangle,
                                 const size_t,
                                 const float2,
                                 const cl_mem, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hpmv<double2>(const Layout, const Triangle,
                                  const size_t,
                                  const double2,
                                  const cl_mem, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV
template <typename T>
StatusCode Symv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsymv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSymv(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Symv<float>(const Layout, const Triangle,
                                const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Symv<double>(const Layout, const Triangle,
                                 const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV
template <typename T>
StatusCode Sbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsbmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSbmv(layout, triangle,
                        n, k,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Sbmv<float>(const Layout, const Triangle,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Sbmv<double>(const Layout, const Triangle,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV
template <typename T>
StatusCode Spmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem ap_buffer, const size_t ap_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xspmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSpmv(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(ap_buffer), ap_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        beta,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode Spmv<float>(const Layout, const Triangle,
                                const size_t,
                                const float,
                                const cl_mem, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Spmv<double>(const Layout, const Triangle,
                                 const size_t,
                                 const double,
                                 const cl_mem, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV
template <typename T>
StatusCode Trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtrmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTrmv(layout, triangle, a_transpose, diagonal,
                        n,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode Trmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Trmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                  const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV
template <typename T>
StatusCode Tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n, const size_t k,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtbmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTbmv(layout, triangle, a_transpose, diagonal,
                        n, k,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode Tbmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Tbmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tbmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tbmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                  const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV
template <typename T>
StatusCode Tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem ap_buffer, const size_t ap_offset,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtpmv<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTpmv(layout, triangle, a_transpose, diagonal,
                        n,
                        Buffer<T>(ap_buffer), ap_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode Tpmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                const size_t,
                                const cl_mem, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Tpmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tpmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tpmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                  const size_t,
                                  const cl_mem, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
StatusCode Trsv(const Layout, const Triangle, const Transpose, const Diagonal,
                const size_t,
                const cl_mem, const size_t, const size_t,
                cl_mem, const size_t, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode Trsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Trsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                  const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
StatusCode Tbsv(const Layout, const Triangle, const Transpose, const Diagonal,
                const size_t, const size_t,
                const cl_mem, const size_t, const size_t,
                cl_mem, const size_t, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode Tbsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Tbsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tbsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tbsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                  const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
StatusCode Tpsv(const Layout, const Triangle, const Transpose, const Diagonal,
                const size_t,
                const cl_mem, const size_t,
                cl_mem, const size_t, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode Tpsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                const size_t,
                                const cl_mem, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Tpsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tpsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                 const size_t,
                                 const cl_mem, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Tpsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                  const size_t,
                                  const cl_mem, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// General rank-1 matrix update: SGER/DGER
template <typename T>
StatusCode Ger(const Layout layout,
               const size_t m, const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xger<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGer(layout,
                       m, n,
                       alpha,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(y_buffer), y_offset, y_inc,
                       Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode Ger<float>(const Layout,
                               const size_t, const size_t,
                               const float,
                               const cl_mem, const size_t, const size_t,
                               const cl_mem, const size_t, const size_t,
                               cl_mem, const size_t, const size_t,
                               cl_command_queue*, cl_event*);
template StatusCode Ger<double>(const Layout,
                                const size_t, const size_t,
                                const double,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);

// General rank-1 complex matrix update: CGERU/ZGERU
template <typename T>
StatusCode Geru(const Layout layout,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgeru<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGeru(layout,
                        m, n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode Geru<float2>(const Layout,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Geru<double2>(const Layout,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
template <typename T>
StatusCode Gerc(const Layout layout,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgerc<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGerc(layout,
                        m, n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode Gerc<float2>(const Layout,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gerc<double2>(const Layout,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Hermitian rank-1 matrix update: CHER/ZHER
template <typename T>
StatusCode Her(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xher<std::complex<T>,T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHer(layout, triangle,
                       n,
                       alpha,
                       Buffer<std::complex<T>>(x_buffer), x_offset, x_inc,
                       Buffer<std::complex<T>>(a_buffer), a_offset, a_ld);
}
template StatusCode Her<float>(const Layout, const Triangle,
                               const size_t,
                               const float,
                               const cl_mem, const size_t, const size_t,
                               cl_mem, const size_t, const size_t,
                               cl_command_queue*, cl_event*);
template StatusCode Her<double>(const Layout, const Triangle,
                                const size_t,
                                const double,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
template <typename T>
StatusCode Hpr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem ap_buffer, const size_t ap_offset,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhpr<std::complex<T>,T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHpr(layout, triangle,
                       n,
                       alpha,
                       Buffer<std::complex<T>>(x_buffer), x_offset, x_inc,
                       Buffer<std::complex<T>>(ap_buffer), ap_offset);
}
template StatusCode Hpr<float>(const Layout, const Triangle,
                               const size_t,
                               const float,
                               const cl_mem, const size_t, const size_t,
                               cl_mem, const size_t,
                               cl_command_queue*, cl_event*);
template StatusCode Hpr<double>(const Layout, const Triangle,
                                const size_t,
                                const double,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t,
                                cl_command_queue*, cl_event*);

// Hermitian rank-2 matrix update: CHER2/ZHER2
template <typename T>
StatusCode Her2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xher2<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHer2(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode Her2<float2>(const Layout, const Triangle,
                                 const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Her2<double2>(const Layout, const Triangle,
                                  const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
StatusCode Hpr2(const Layout, const Triangle,
                const size_t,
                const T,
                const cl_mem, const size_t, const size_t,
                const cl_mem, const size_t, const size_t,
                cl_mem, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode Hpr2<float2>(const Layout, const Triangle,
                                 const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hpr2<double2>(const Layout, const Triangle,
                                  const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t,
                                  cl_command_queue*, cl_event*);

// Symmetric rank-1 matrix update: SSYR/DSYR
template <typename T>
StatusCode Syr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyr<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyr(layout, triangle,
                       n,
                       alpha,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode Syr<float>(const Layout, const Triangle,
                               const size_t,
                               const float,
                               const cl_mem, const size_t, const size_t,
                               cl_mem, const size_t, const size_t,
                               cl_command_queue*, cl_event*);
template StatusCode Syr<double>(const Layout, const Triangle,
                                const size_t,
                                const double,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);

// Symmetric packed rank-1 matrix update: SSPR/DSPR
template <typename T>
StatusCode Spr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem ap_buffer, const size_t ap_offset,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xspr<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSpr(layout, triangle,
                       n,
                       alpha,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(ap_buffer), ap_offset);
}
template StatusCode Spr<float>(const Layout, const Triangle,
                               const size_t,
                               const float,
                               const cl_mem, const size_t, const size_t,
                               cl_mem, const size_t,
                               cl_command_queue*, cl_event*);
template StatusCode Spr<double>(const Layout, const Triangle,
                                const size_t,
                                const double,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t,
                                cl_command_queue*, cl_event*);

// Symmetric rank-2 matrix update: SSYR2/DSYR2
template <typename T>
StatusCode Syr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyr2<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyr2(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode Syr2<float>(const Layout, const Triangle,
                                const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Syr2<double>(const Layout, const Triangle,
                                 const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2
template <typename T>
StatusCode Spr2(const Layout, const Triangle,
                const size_t,
                const T,
                const cl_mem, const size_t, const size_t,
                const cl_mem, const size_t, const size_t,
                cl_mem, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode Spr2<float>(const Layout, const Triangle,
                                const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Spr2<double>(const Layout, const Triangle,
                                 const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t,
                                 cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                const size_t m, const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xgemm<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGemm(layout, a_transpose, b_transpose,
                        m, n, k,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld,
                        beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Gemm<float>(const Layout, const Transpose, const Transpose,
                                const size_t, const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Gemm<double>(const Layout, const Transpose, const Transpose,
                                 const size_t, const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemm<float2>(const Layout, const Transpose, const Transpose,
                                 const size_t, const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Gemm<double2>(const Layout, const Transpose, const Transpose,
                                  const size_t, const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM
template <typename T>
StatusCode Symm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsymm<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSymm(layout, side, triangle,
                        m, n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld,
                        beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Symm<float>(const Layout, const Side, const Triangle,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Symm<double>(const Layout, const Side, const Triangle,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Symm<float2>(const Layout, const Side, const Triangle,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Symm<double2>(const Layout, const Side, const Triangle,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
template <typename T>
StatusCode Hemm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xhemm<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHemm(layout, side, triangle,
                        m, n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld,
                        beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Hemm<float2>(const Layout, const Side, const Triangle,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Hemm<double2>(const Layout, const Side, const Triangle,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK
template <typename T>
StatusCode Syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyrk<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyrk(layout, triangle, a_transpose,
                        n, k,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Syrk<float>(const Layout, const Triangle, const Transpose,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Syrk<double>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Syrk<float2>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 const float2,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Syrk<double2>(const Layout, const Triangle, const Transpose,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  const double2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
template <typename T>
StatusCode Herk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xherk<std::complex<T>,T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHerk(layout, triangle, a_transpose,
                        n, k,
                        alpha,
                        Buffer<std::complex<T>>(a_buffer), a_offset, a_ld,
                        beta,
                        Buffer<std::complex<T>>(c_buffer), c_offset, c_ld);
}
template StatusCode Herk<float>(const Layout, const Triangle, const Transpose,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                const float,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Herk<double>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 const double,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K
template <typename T>
StatusCode Syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                 const T beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xsyr2k<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyr2k(layout, triangle, ab_transpose,
                         n, k,
                         alpha,
                         Buffer<T>(a_buffer), a_offset, a_ld,
                         Buffer<T>(b_buffer), b_offset, b_ld,
                         beta,
                         Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Syr2k<float>(const Layout, const Triangle, const Transpose,
                                 const size_t, const size_t,
                                 const float,
                                 const cl_mem, const size_t, const size_t,
                                 const cl_mem, const size_t, const size_t,
                                 const float,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Syr2k<double>(const Layout, const Triangle, const Transpose,
                                  const size_t, const size_t,
                                  const double,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const double,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);
template StatusCode Syr2k<float2>(const Layout, const Triangle, const Transpose,
                                  const size_t, const size_t,
                                  const float2,
                                  const cl_mem, const size_t, const size_t,
                                  const cl_mem, const size_t, const size_t,
                                  const float2,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);
template StatusCode Syr2k<double2>(const Layout, const Triangle, const Transpose,
                                   const size_t, const size_t,
                                   const double2,
                                   const cl_mem, const size_t, const size_t,
                                   const cl_mem, const size_t, const size_t,
                                   const double2,
                                   cl_mem, const size_t, const size_t,
                                   cl_command_queue*, cl_event*);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
template <typename T, typename U>
StatusCode Her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                 const U beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xher2k<T,U>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHer2k(layout, triangle, ab_transpose,
                         n, k,
                         alpha,
                         Buffer<T>(a_buffer), a_offset, a_ld,
                         Buffer<T>(b_buffer), b_offset, b_ld,
                         beta,
                         Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode Her2k<float2,float>(const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t,
                                        const float2,
                                        const cl_mem, const size_t, const size_t,
                                        const cl_mem, const size_t, const size_t,
                                        const float,
                                        cl_mem, const size_t, const size_t,
                                        cl_command_queue*, cl_event*);
template StatusCode Her2k<double2,double>(const Layout, const Triangle, const Transpose,
                                          const size_t, const size_t,
                                          const double2,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const double,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM
template <typename T>
StatusCode Trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto event_cpp = Event(*event);
  auto routine = Xtrmm<T>(queue_cpp, event_cpp);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTrmm(layout, side, triangle, a_transpose, diagonal,
                        m, n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld);
}
template StatusCode Trmm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Trmm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trmm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trmm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
StatusCode Trsm(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                const size_t, const size_t,
                const T,
                const cl_mem, const size_t, const size_t,
                cl_mem, const size_t, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode Trsm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                const size_t, const size_t,
                                const float,
                                const cl_mem, const size_t, const size_t,
                                cl_mem, const size_t, const size_t,
                                cl_command_queue*, cl_event*);
template StatusCode Trsm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const double,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trsm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                 const size_t, const size_t,
                                 const float2,
                                 const cl_mem, const size_t, const size_t,
                                 cl_mem, const size_t, const size_t,
                                 cl_command_queue*, cl_event*);
template StatusCode Trsm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                  const size_t, const size_t,
                                  const double2,
                                  const cl_mem, const size_t, const size_t,
                                  cl_mem, const size_t, const size_t,
                                  cl_command_queue*, cl_event*);

// =================================================================================================
} // namespace clblast
