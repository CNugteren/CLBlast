
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
#include "internal/public_api.h"
#include "internal/cache.h"

// BLAS level-1 includes
#include "internal/routines/level1/xswap.h"
#include "internal/routines/level1/xscal.h"
#include "internal/routines/level1/xcopy.h"
#include "internal/routines/level1/xaxpy.h"
#include "internal/routines/level1/xdot.h"
#include "internal/routines/level1/xdotu.h"
#include "internal/routines/level1/xdotc.h"
#include "internal/routines/level1/xnrm2.h"
#include "internal/routines/level1/xasum.h"
#include "internal/routines/level1/xsum.h" // non-BLAS function
#include "internal/routines/level1/xamax.h"
#include "internal/routines/level1/xmax.h" // non-BLAS function

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
#include "internal/routines/level2/xhpr2.h"
#include "internal/routines/level2/xsyr.h"
#include "internal/routines/level2/xspr.h"
#include "internal/routines/level2/xsyr2.h"
#include "internal/routines/level2/xspr2.h"

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

// Generate givens plane rotation: SROTG/DROTG
template <typename T>
StatusCode Rotg(cl_mem, const size_t,
                cl_mem, const size_t,
                cl_mem, const size_t,
                cl_mem, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rotg<float>(cl_mem, const size_t,
                                           cl_mem, const size_t,
                                           cl_mem, const size_t,
                                           cl_mem, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Rotg<double>(cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            cl_command_queue*, cl_event*);

// Generate modified givens plane rotation: SROTMG/DROTMG
template <typename T>
StatusCode Rotmg(cl_mem, const size_t,
                 cl_mem, const size_t,
                 cl_mem, const size_t,
                 const cl_mem, const size_t,
                 cl_mem, const size_t,
                 cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rotmg<float>(cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Rotmg<double>(cl_mem, const size_t,
                                             cl_mem, const size_t,
                                             cl_mem, const size_t,
                                             const cl_mem, const size_t,
                                             cl_mem, const size_t,
                                             cl_command_queue*, cl_event*);

// Apply givens plane rotation: SROT/DROT
template <typename T>
StatusCode Rot(const size_t,
               cl_mem, const size_t, const size_t,
               cl_mem, const size_t, const size_t,
               const T,
               const T,
               cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rot<float>(const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          const float,
                                          const float,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Rot<double>(const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           const double,
                                           const double,
                                           cl_command_queue*, cl_event*);

// Apply modified givens plane rotation: SROTM/DROTM
template <typename T>
StatusCode Rotm(const size_t,
                cl_mem, const size_t, const size_t,
                cl_mem, const size_t, const size_t,
                cl_mem, const size_t,
                cl_command_queue*, cl_event*) {
  return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rotm<float>(const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Rotm<double>(const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t,
                                            cl_command_queue*, cl_event*);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP
template <typename T>
StatusCode Swap(const size_t n,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xswap<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSwap(n,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode PUBLIC_API Swap<float>(const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Swap<double>(const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Swap<float2>(const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Swap<double2>(const size_t,
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
  auto routine = Xscal<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoScal(n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Scal<float>(const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Scal<double>(const size_t,
                                            const double,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Scal<float2>(const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Scal<double2>(const size_t,
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
  auto routine = Xcopy<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoCopy(n,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode PUBLIC_API Copy<float>(const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Copy<double>(const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Copy<float2>(const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Copy<double2>(const size_t,
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
  auto routine = Xaxpy<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoAxpy(n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode PUBLIC_API Axpy<float>(const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Axpy<double>(const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Axpy<float2>(const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Axpy<double2>(const size_t,
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
  auto routine = Xdot<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoDot(n,
                       Buffer<T>(dot_buffer), dot_offset,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode PUBLIC_API Dot<float>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Dot<double>(const size_t,
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
  auto routine = Xdotu<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoDotu(n,
                        Buffer<T>(dot_buffer), dot_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode PUBLIC_API Dotu<float2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Dotu<double2>(const size_t,
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
  auto routine = Xdotc<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoDotc(n,
                        Buffer<T>(dot_buffer), dot_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc);
}
template StatusCode PUBLIC_API Dotc<float2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Dotc<double2>(const size_t,
                                             cl_mem, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2
template <typename T>
StatusCode Nrm2(const size_t n,
                cl_mem nrm2_buffer, const size_t nrm2_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xnrm2<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoNrm2(n,
                        Buffer<T>(nrm2_buffer), nrm2_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Nrm2<float>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Nrm2<double>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Nrm2<float2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Nrm2<double2>(const size_t,
                                             cl_mem, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM
template <typename T>
StatusCode Asum(const size_t n,
                cl_mem asum_buffer, const size_t asum_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xasum<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoAsum(n,
                        Buffer<T>(asum_buffer), asum_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Asum<float>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Asum<double>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Asum<float2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Asum<double2>(const size_t,
                                             cl_mem, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM
template <typename T>
StatusCode Sum(const size_t n,
               cl_mem sum_buffer, const size_t sum_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xsum<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSum(n,
                       Buffer<T>(sum_buffer), sum_offset,
                       Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Sum<float>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Sum<double>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Sum<float2>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Sum<double2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX
template <typename T>
StatusCode Amax(const size_t n,
                cl_mem imax_buffer, const size_t imax_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xamax<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoAmax(n,
                        Buffer<T>(imax_buffer), imax_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Amax<float>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amax<double>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amax<float2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amax<double2>(const size_t,
                                             cl_mem, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX
template <typename T>
StatusCode Max(const size_t n,
               cl_mem imax_buffer, const size_t imax_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xmax<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoMax(n,
                       Buffer<T>(imax_buffer), imax_offset,
                       Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Max<float>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Max<double>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Max<float2>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Max<double2>(const size_t,
                                            cl_mem, const size_t,
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
  auto routine = Xgemv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Gemv<float>(const Layout, const Transpose,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gemv<double>(const Layout, const Transpose,
                                            const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const double,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gemv<float2>(const Layout, const Transpose,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gemv<double2>(const Layout, const Transpose,
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
  auto routine = Xgbmv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Gbmv<float>(const Layout, const Transpose,
                                           const size_t, const size_t, const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gbmv<double>(const Layout, const Transpose,
                                            const size_t, const size_t, const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const double,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gbmv<float2>(const Layout, const Transpose,
                                            const size_t, const size_t, const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gbmv<double2>(const Layout, const Transpose,
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
  auto routine = Xhemv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Hemv<float2>(const Layout, const Triangle,
                                            const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Hemv<double2>(const Layout, const Triangle,
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
  auto routine = Xhbmv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Hbmv<float2>(const Layout, const Triangle,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Hbmv<double2>(const Layout, const Triangle,
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
  auto routine = Xhpmv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Hpmv<float2>(const Layout, const Triangle,
                                            const size_t,
                                            const float2,
                                            const cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Hpmv<double2>(const Layout, const Triangle,
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
  auto routine = Xsymv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Symv<float>(const Layout, const Triangle,
                                           const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Symv<double>(const Layout, const Triangle,
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
  auto routine = Xsbmv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Sbmv<float>(const Layout, const Triangle,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Sbmv<double>(const Layout, const Triangle,
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
  auto routine = Xspmv<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Spmv<float>(const Layout, const Triangle,
                                           const size_t,
                                           const float,
                                           const cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Spmv<double>(const Layout, const Triangle,
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
  auto routine = Xtrmv<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTrmv(layout, triangle, a_transpose, diagonal,
                        n,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Trmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                           const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
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
  auto routine = Xtbmv<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTbmv(layout, triangle, a_transpose, diagonal,
                        n, k,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Tbmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                           const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tbmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tbmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tbmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
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
  auto routine = Xtpmv<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTpmv(layout, triangle, a_transpose, diagonal,
                        n,
                        Buffer<T>(ap_buffer), ap_offset,
                        Buffer<T>(x_buffer), x_offset, x_inc);
}
template StatusCode PUBLIC_API Tpmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                           const size_t,
                                           const cl_mem, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tpmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tpmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tpmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
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
template StatusCode PUBLIC_API Trsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                           const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
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
template StatusCode PUBLIC_API Tbsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                           const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tbsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tbsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tbsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
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
template StatusCode PUBLIC_API Tpsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                           const size_t,
                                           const cl_mem, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tpsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tpsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                            const size_t,
                                            const cl_mem, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Tpsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
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
  auto routine = Xger<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGer(layout,
                       m, n,
                       alpha,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(y_buffer), y_offset, y_inc,
                       Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Ger<float>(const Layout,
                                          const size_t, const size_t,
                                          const float,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Ger<double>(const Layout,
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
  auto routine = Xgeru<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGeru(layout,
                        m, n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Geru<float2>(const Layout,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Geru<double2>(const Layout,
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
  auto routine = Xgerc<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoGerc(layout,
                        m, n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Gerc<float2>(const Layout,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gerc<double2>(const Layout,
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
  auto routine = Xher<std::complex<T>,T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHer(layout, triangle,
                       n,
                       alpha,
                       Buffer<std::complex<T>>(x_buffer), x_offset, x_inc,
                       Buffer<std::complex<T>>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Her<float>(const Layout, const Triangle,
                                          const size_t,
                                          const float,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Her<double>(const Layout, const Triangle,
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
  auto routine = Xhpr<std::complex<T>,T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHpr(layout, triangle,
                       n,
                       alpha,
                       Buffer<std::complex<T>>(x_buffer), x_offset, x_inc,
                       Buffer<std::complex<T>>(ap_buffer), ap_offset);
}
template StatusCode PUBLIC_API Hpr<float>(const Layout, const Triangle,
                                          const size_t,
                                          const float,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Hpr<double>(const Layout, const Triangle,
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
  auto routine = Xher2<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHer2(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Her2<float2>(const Layout, const Triangle,
                                            const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Her2<double2>(const Layout, const Triangle,
                                             const size_t,
                                             const double2,
                                             const cl_mem, const size_t, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
StatusCode Hpr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem ap_buffer, const size_t ap_offset,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xhpr2<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHpr2(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(ap_buffer), ap_offset);
}
template StatusCode PUBLIC_API Hpr2<float2>(const Layout, const Triangle,
                                            const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Hpr2<double2>(const Layout, const Triangle,
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
  auto routine = Xsyr<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyr(layout, triangle,
                       n,
                       alpha,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Syr<float>(const Layout, const Triangle,
                                          const size_t,
                                          const float,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syr<double>(const Layout, const Triangle,
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
  auto routine = Xspr<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSpr(layout, triangle,
                       n,
                       alpha,
                       Buffer<T>(x_buffer), x_offset, x_inc,
                       Buffer<T>(ap_buffer), ap_offset);
}
template StatusCode PUBLIC_API Spr<float>(const Layout, const Triangle,
                                          const size_t,
                                          const float,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Spr<double>(const Layout, const Triangle,
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
  auto routine = Xsyr2<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyr2(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(a_buffer), a_offset, a_ld);
}
template StatusCode PUBLIC_API Syr2<float>(const Layout, const Triangle,
                                           const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syr2<double>(const Layout, const Triangle,
                                            const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2
template <typename T>
StatusCode Spr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem ap_buffer, const size_t ap_offset,
                cl_command_queue* queue, cl_event* event) {
  auto queue_cpp = Queue(*queue);
  auto routine = Xspr2<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSpr2(layout, triangle,
                        n,
                        alpha,
                        Buffer<T>(x_buffer), x_offset, x_inc,
                        Buffer<T>(y_buffer), y_offset, y_inc,
                        Buffer<T>(ap_buffer), ap_offset);
}
template StatusCode PUBLIC_API Spr2<float>(const Layout, const Triangle,
                                           const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Spr2<double>(const Layout, const Triangle,
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
  auto routine = Xgemm<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Gemm<float>(const Layout, const Transpose, const Transpose,
                                           const size_t, const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gemm<double>(const Layout, const Transpose, const Transpose,
                                            const size_t, const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const double,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gemm<float2>(const Layout, const Transpose, const Transpose,
                                            const size_t, const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Gemm<double2>(const Layout, const Transpose, const Transpose,
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
  auto routine = Xsymm<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Symm<float>(const Layout, const Side, const Triangle,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Symm<double>(const Layout, const Side, const Triangle,
                                            const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const double,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Symm<float2>(const Layout, const Side, const Triangle,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Symm<double2>(const Layout, const Side, const Triangle,
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
  auto routine = Xhemm<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Hemm<float2>(const Layout, const Side, const Triangle,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Hemm<double2>(const Layout, const Side, const Triangle,
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
  auto routine = Xsyrk<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoSyrk(layout, triangle, a_transpose,
                        n, k,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        beta,
                        Buffer<T>(c_buffer), c_offset, c_ld);
}
template StatusCode PUBLIC_API Syrk<float>(const Layout, const Triangle, const Transpose,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syrk<double>(const Layout, const Triangle, const Transpose,
                                            const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            const double,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syrk<float2>(const Layout, const Triangle, const Transpose,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            const float2,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syrk<double2>(const Layout, const Triangle, const Transpose,
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
  auto routine = Xherk<std::complex<T>,T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoHerk(layout, triangle, a_transpose,
                        n, k,
                        alpha,
                        Buffer<std::complex<T>>(a_buffer), a_offset, a_ld,
                        beta,
                        Buffer<std::complex<T>>(c_buffer), c_offset, c_ld);
}
template StatusCode PUBLIC_API Herk<float>(const Layout, const Triangle, const Transpose,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           const float,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Herk<double>(const Layout, const Triangle, const Transpose,
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
  auto routine = Xsyr2k<T>(queue_cpp, event);
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
template StatusCode PUBLIC_API Syr2k<float>(const Layout, const Triangle, const Transpose,
                                            const size_t, const size_t,
                                            const float,
                                            const cl_mem, const size_t, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            const float,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syr2k<double>(const Layout, const Triangle, const Transpose,
                                             const size_t, const size_t,
                                             const double,
                                             const cl_mem, const size_t, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             const double,
                                             cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syr2k<float2>(const Layout, const Triangle, const Transpose,
                                             const size_t, const size_t,
                                             const float2,
                                             const cl_mem, const size_t, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             const float2,
                                             cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Syr2k<double2>(const Layout, const Triangle, const Transpose,
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
  auto routine = Xher2k<T,U>(queue_cpp, event);
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
template StatusCode PUBLIC_API Her2k<float2,float>(const Layout, const Triangle, const Transpose,
                                                   const size_t, const size_t,
                                                   const float2,
                                                   const cl_mem, const size_t, const size_t,
                                                   const cl_mem, const size_t, const size_t,
                                                   const float,
                                                   cl_mem, const size_t, const size_t,
                                                   cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Her2k<double2,double>(const Layout, const Triangle, const Transpose,
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
  auto routine = Xtrmm<T>(queue_cpp, event);
  auto status = routine.SetUp();
  if (status != StatusCode::kSuccess) { return status; }
  return routine.DoTrmm(layout, side, triangle, a_transpose, diagonal,
                        m, n,
                        alpha,
                        Buffer<T>(a_buffer), a_offset, a_ld,
                        Buffer<T>(b_buffer), b_offset, b_ld);
}
template StatusCode PUBLIC_API Trmm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trmm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trmm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trmm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
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
template StatusCode PUBLIC_API Trsm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                           const size_t, const size_t,
                                           const float,
                                           const cl_mem, const size_t, const size_t,
                                           cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trsm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const double,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trsm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                            const size_t, const size_t,
                                            const float2,
                                            const cl_mem, const size_t, const size_t,
                                            cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Trsm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                             const size_t, const size_t,
                                             const double2,
                                             const cl_mem, const size_t, const size_t,
                                             cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);

// =================================================================================================

// Clears the cache of stored binaries
StatusCode ClearCache() { return cache::ClearCache(); }

// =================================================================================================
} // namespace clblast
