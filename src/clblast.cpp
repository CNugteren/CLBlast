
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

#include "cache.hpp"
#include "clblast.h"

// BLAS level-1 includes
#include "routines/level1/xswap.hpp"
#include "routines/level1/xscal.hpp"
#include "routines/level1/xcopy.hpp"
#include "routines/level1/xaxpy.hpp"
#include "routines/level1/xdot.hpp"
#include "routines/level1/xdotu.hpp"
#include "routines/level1/xdotc.hpp"
#include "routines/level1/xnrm2.hpp"
#include "routines/level1/xasum.hpp"
#include "routines/level1/xsum.hpp" // non-BLAS routine
#include "routines/level1/xamax.hpp"
#include "routines/level1/xamin.hpp" // non-BLAS routine
#include "routines/level1/xmax.hpp" // non-BLAS routine
#include "routines/level1/xmin.hpp" // non-BLAS routine

// BLAS level-2 includes
#include "routines/level2/xgemv.hpp"
#include "routines/level2/xgbmv.hpp"
#include "routines/level2/xhemv.hpp"
#include "routines/level2/xhbmv.hpp"
#include "routines/level2/xhpmv.hpp"
#include "routines/level2/xsymv.hpp"
#include "routines/level2/xsbmv.hpp"
#include "routines/level2/xspmv.hpp"
#include "routines/level2/xtrmv.hpp"
#include "routines/level2/xtbmv.hpp"
#include "routines/level2/xtpmv.hpp"
#include "routines/level2/xtrsv.hpp"
#include "routines/level2/xger.hpp"
#include "routines/level2/xgeru.hpp"
#include "routines/level2/xgerc.hpp"
#include "routines/level2/xher.hpp"
#include "routines/level2/xhpr.hpp"
#include "routines/level2/xher2.hpp"
#include "routines/level2/xhpr2.hpp"
#include "routines/level2/xsyr.hpp"
#include "routines/level2/xspr.hpp"
#include "routines/level2/xsyr2.hpp"
#include "routines/level2/xspr2.hpp"

// BLAS level-3 includes
#include "routines/level3/xgemm.hpp"
#include "routines/level3/xsymm.hpp"
#include "routines/level3/xhemm.hpp"
#include "routines/level3/xsyrk.hpp"
#include "routines/level3/xherk.hpp"
#include "routines/level3/xsyr2k.hpp"
#include "routines/level3/xher2k.hpp"
#include "routines/level3/xtrmm.hpp"
#include "routines/level3/xtrsm.hpp"

// Level-x includes (non-BLAS)
#include "routines/levelx/xomatcopy.hpp"
#include "routines/levelx/xim2col.hpp"
#include "routines/levelx/xaxpybatched.hpp"
#include "routines/levelx/xgemmbatched.hpp"

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

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
template <typename T>
StatusCode Swap(const size_t n,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xswap<T>(queue_cpp, event);
    routine.DoSwap(n,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Swap<half>(const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
template <typename T>
StatusCode Scal(const size_t n,
                const T alpha,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xscal<T>(queue_cpp, event);
    routine.DoScal(n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Scal<half>(const size_t,
                                          const half,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
template <typename T>
StatusCode Copy(const size_t n,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xcopy<T>(queue_cpp, event);
    routine.DoCopy(n,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Copy<half>(const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
template <typename T>
StatusCode Axpy(const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xaxpy<T>(queue_cpp, event);
    routine.DoAxpy(n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Axpy<half>(const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Dot product of two vectors: SDOT/DDOT/HDOT
template <typename T>
StatusCode Dot(const size_t n,
               cl_mem dot_buffer, const size_t dot_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xdot<T>(queue_cpp, event);
    routine.DoDot(n,
                  Buffer<T>(dot_buffer), dot_offset,
                  Buffer<T>(x_buffer), x_offset, x_inc,
                  Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Dot<half>(const size_t,
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xdotu<T>(queue_cpp, event);
    routine.DoDotu(n,
                   Buffer<T>(dot_buffer), dot_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xdotc<T>(queue_cpp, event);
    routine.DoDotc(n,
                   Buffer<T>(dot_buffer), dot_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
template <typename T>
StatusCode Nrm2(const size_t n,
                cl_mem nrm2_buffer, const size_t nrm2_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xnrm2<T>(queue_cpp, event);
    routine.DoNrm2(n,
                   Buffer<T>(nrm2_buffer), nrm2_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Nrm2<half>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
template <typename T>
StatusCode Asum(const size_t n,
                cl_mem asum_buffer, const size_t asum_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xasum<T>(queue_cpp, event);
    routine.DoAsum(n,
                   Buffer<T>(asum_buffer), asum_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Asum<half>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
template <typename T>
StatusCode Sum(const size_t n,
               cl_mem sum_buffer, const size_t sum_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsum<T>(queue_cpp, event);
    routine.DoSum(n,
                  Buffer<T>(sum_buffer), sum_offset,
                  Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Sum<half>(const size_t,
                                         cl_mem, const size_t,
                                         const cl_mem, const size_t, const size_t,
                                         cl_command_queue*, cl_event*);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
StatusCode Amax(const size_t n,
                cl_mem imax_buffer, const size_t imax_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xamax<T>(queue_cpp, event);
    routine.DoAmax(n,
                   Buffer<unsigned int>(imax_buffer), imax_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Amax<half>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
template <typename T>
StatusCode Amin(const size_t n,
                cl_mem imin_buffer, const size_t imin_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xamin<T>(queue_cpp, event);
    routine.DoAmin(n,
                   Buffer<unsigned int>(imin_buffer), imin_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
}
template StatusCode PUBLIC_API Amin<float>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amin<double>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amin<float2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amin<double2>(const size_t,
                                             cl_mem, const size_t,
                                             const cl_mem, const size_t, const size_t,
                                             cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Amin<half>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
template <typename T>
StatusCode Max(const size_t n,
               cl_mem imax_buffer, const size_t imax_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xmax<T>(queue_cpp, event);
    routine.DoMax(n,
                  Buffer<unsigned int>(imax_buffer), imax_offset,
                  Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Max<half>(const size_t,
                                         cl_mem, const size_t,
                                         const cl_mem, const size_t, const size_t,
                                         cl_command_queue*, cl_event*);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
template <typename T>
StatusCode Min(const size_t n,
               cl_mem imin_buffer, const size_t imin_offset,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xmin<T>(queue_cpp, event);
    routine.DoMin(n,
                  Buffer<unsigned int>(imin_buffer), imin_offset,
                  Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
}
template StatusCode PUBLIC_API Min<float>(const size_t,
                                          cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Min<double>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Min<float2>(const size_t,
                                           cl_mem, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Min<double2>(const size_t,
                                            cl_mem, const size_t,
                                            const cl_mem, const size_t, const size_t,
                                            cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Min<half>(const size_t,
                                         cl_mem, const size_t,
                                         const cl_mem, const size_t, const size_t,
                                         cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xgemv<T>(queue_cpp, event);
    routine.DoGemv(layout, a_transpose,
                   m, n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Gemv<half>(const Layout, const Transpose,
                                          const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
template <typename T>
StatusCode Gbmv(const Layout layout, const Transpose a_transpose,
                const size_t m, const size_t n, const size_t kl, const size_t ku,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xgbmv<T>(queue_cpp, event);
    routine.DoGbmv(layout, a_transpose,
                   m, n, kl, ku,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Gbmv<half>(const Layout, const Transpose,
                                          const size_t, const size_t, const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xhemv<T>(queue_cpp, event);
    routine.DoHemv(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xhbmv<T>(queue_cpp, event);
    routine.DoHbmv(layout, triangle,
                   n, k,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xhpmv<T>(queue_cpp, event);
    routine.DoHpmv(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(ap_buffer), ap_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
template <typename T>
StatusCode Symv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsymv<T>(queue_cpp, event);
    routine.DoSymv(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Symv<half>(const Layout, const Triangle,
                                          const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
template <typename T>
StatusCode Sbmv(const Layout layout, const Triangle triangle,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsbmv<T>(queue_cpp, event);
    routine.DoSbmv(layout, triangle,
                   n, k,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Sbmv<half>(const Layout, const Triangle,
                                          const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
template <typename T>
StatusCode Spmv(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem ap_buffer, const size_t ap_offset,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const T beta,
                cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xspmv<T>(queue_cpp, event);
    routine.DoSpmv(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(ap_buffer), ap_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   beta,
                   Buffer<T>(y_buffer), y_offset, y_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Spmv<half>(const Layout, const Triangle,
                                          const size_t,
                                          const half,
                                          const cl_mem, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
template <typename T>
StatusCode Trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xtrmv<T>(queue_cpp, event);
    routine.DoTrmv(layout, triangle, a_transpose, diagonal,
                   n,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Trmv<half>(const Layout, const Triangle, const Transpose, const Diagonal,
                                          const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
template <typename T>
StatusCode Tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n, const size_t k,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xtbmv<T>(queue_cpp, event);
    routine.DoTbmv(layout, triangle, a_transpose, diagonal,
                   n, k,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Tbmv<half>(const Layout, const Triangle, const Transpose, const Diagonal,
                                          const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
template <typename T>
StatusCode Tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem ap_buffer, const size_t ap_offset,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xtpmv<T>(queue_cpp, event);
    routine.DoTpmv(layout, triangle, a_transpose, diagonal,
                   n,
                   Buffer<T>(ap_buffer), ap_offset,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Tpmv<half>(const Layout, const Triangle, const Transpose, const Diagonal,
                                          const size_t,
                                          const cl_mem, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
StatusCode Trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xtrsv<T>(queue_cpp, event);
    routine.DoTrsv(layout, triangle, a_transpose, diagonal,
                   n,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(x_buffer), x_offset, x_inc);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// General rank-1 matrix update: SGER/DGER/HGER
template <typename T>
StatusCode Ger(const Layout layout,
               const size_t m, const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xger<T>(queue_cpp, event);
    routine.DoGer(layout,
                  m, n,
                  alpha,
                  Buffer<T>(x_buffer), x_offset, x_inc,
                  Buffer<T>(y_buffer), y_offset, y_inc,
                  Buffer<T>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Ger<half>(const Layout,
                                         const size_t, const size_t,
                                         const half,
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xgeru<T>(queue_cpp, event);
    routine.DoGeru(layout,
                   m, n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc,
                   Buffer<T>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xgerc<T>(queue_cpp, event);
    routine.DoGerc(layout,
                   m, n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc,
                   Buffer<T>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xher<std::complex<T>,T>(queue_cpp, event);
    routine.DoHer(layout, triangle,
                  n,
                  alpha,
                  Buffer<std::complex<T>>(x_buffer), x_offset, x_inc,
                  Buffer<std::complex<T>>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xhpr<std::complex<T>,T>(queue_cpp, event);
    routine.DoHpr(layout, triangle,
                  n,
                  alpha,
                  Buffer<std::complex<T>>(x_buffer), x_offset, x_inc,
                  Buffer<std::complex<T>>(ap_buffer), ap_offset);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xher2<T>(queue_cpp, event);
    routine.DoHer2(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc,
                   Buffer<T>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xhpr2<T>(queue_cpp, event);
    routine.DoHpr2(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc,
                   Buffer<T>(ap_buffer), ap_offset);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
template <typename T>
StatusCode Syr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsyr<T>(queue_cpp, event);
    routine.DoSyr(layout, triangle,
                  n,
                  alpha,
                  Buffer<T>(x_buffer), x_offset, x_inc,
                  Buffer<T>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Syr<half>(const Layout, const Triangle,
                                         const size_t,
                                         const half,
                                         const cl_mem, const size_t, const size_t,
                                         cl_mem, const size_t, const size_t,
                                         cl_command_queue*, cl_event*);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
template <typename T>
StatusCode Spr(const Layout layout, const Triangle triangle,
               const size_t n,
               const T alpha,
               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
               cl_mem ap_buffer, const size_t ap_offset,
               cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xspr<T>(queue_cpp, event);
    routine.DoSpr(layout, triangle,
                  n,
                  alpha,
                  Buffer<T>(x_buffer), x_offset, x_inc,
                  Buffer<T>(ap_buffer), ap_offset);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Spr<half>(const Layout, const Triangle,
                                         const size_t,
                                         const half,
                                         const cl_mem, const size_t, const size_t,
                                         cl_mem, const size_t,
                                         cl_command_queue*, cl_event*);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
template <typename T>
StatusCode Syr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsyr2<T>(queue_cpp, event);
    routine.DoSyr2(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc,
                   Buffer<T>(a_buffer), a_offset, a_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Syr2<half>(const Layout, const Triangle,
                                          const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
template <typename T>
StatusCode Spr2(const Layout layout, const Triangle triangle,
                const size_t n,
                const T alpha,
                const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                cl_mem ap_buffer, const size_t ap_offset,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xspr2<T>(queue_cpp, event);
    routine.DoSpr2(layout, triangle,
                   n,
                   alpha,
                   Buffer<T>(x_buffer), x_offset, x_inc,
                   Buffer<T>(y_buffer), y_offset, y_inc,
                   Buffer<T>(ap_buffer), ap_offset);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Spr2<half>(const Layout, const Triangle,
                                          const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t,
                                          cl_command_queue*, cl_event*);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                const size_t m, const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xgemm<T>(queue_cpp, event);
    routine.DoGemm(layout, a_transpose, b_transpose,
                   m, n, k,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(b_buffer), b_offset, b_ld,
                   beta,
                   Buffer<T>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Gemm<half>(const Layout, const Transpose, const Transpose,
                                          const size_t, const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
template <typename T>
StatusCode Symm(const Layout layout, const Side side, const Triangle triangle,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsymm<T>(queue_cpp, event);
    routine.DoSymm(layout, side, triangle,
                   m, n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(b_buffer), b_offset, b_ld,
                   beta,
                   Buffer<T>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Symm<half>(const Layout, const Side, const Triangle,
                                          const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xhemm<T>(queue_cpp, event);
    routine.DoHemm(layout, side, triangle,
                   m, n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(b_buffer), b_offset, b_ld,
                   beta,
                   Buffer<T>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
template <typename T>
StatusCode Syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                const size_t n, const size_t k,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                const T beta,
                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsyrk<T>(queue_cpp, event);
    routine.DoSyrk(layout, triangle, a_transpose,
                   n, k,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   beta,
                   Buffer<T>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Syrk<half>(const Layout, const Triangle, const Transpose,
                                          const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          const half,
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xherk<std::complex<T>,T>(queue_cpp, event);
    routine.DoHerk(layout, triangle, a_transpose,
                   n, k,
                   alpha,
                   Buffer<std::complex<T>>(a_buffer), a_offset, a_ld,
                   beta,
                   Buffer<std::complex<T>>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
template <typename T>
StatusCode Syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                 const size_t n, const size_t k,
                 const T alpha,
                 const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                 const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                 const T beta,
                 cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                 cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xsyr2k<T>(queue_cpp, event);
    routine.DoSyr2k(layout, triangle, ab_transpose,
                    n, k,
                    alpha,
                    Buffer<T>(a_buffer), a_offset, a_ld,
                    Buffer<T>(b_buffer), b_offset, b_ld,
                    beta,
                    Buffer<T>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Syr2k<half>(const Layout, const Triangle, const Transpose,
                                           const size_t, const size_t,
                                           const half,
                                           const cl_mem, const size_t, const size_t,
                                           const cl_mem, const size_t, const size_t,
                                           const half,
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
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xher2k<T,U>(queue_cpp, event);
    routine.DoHer2k(layout, triangle, ab_transpose,
                    n, k,
                    alpha,
                    Buffer<T>(a_buffer), a_offset, a_ld,
                    Buffer<T>(b_buffer), b_offset, b_ld,
                    beta,
                    Buffer<T>(c_buffer), c_offset, c_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
template <typename T>
StatusCode Trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xtrmm<T>(queue_cpp, event);
    routine.DoTrmm(layout, side, triangle, a_transpose, diagonal,
                   m, n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(b_buffer), b_offset, b_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
template StatusCode PUBLIC_API Trmm<half>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                          const size_t, const size_t,
                                          const half,
                                          const cl_mem, const size_t, const size_t,
                                          cl_mem, const size_t, const size_t,
                                          cl_command_queue*, cl_event*);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
StatusCode Trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t m, const size_t n,
                const T alpha,
                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xtrsm<T>(queue_cpp, event);
    routine.DoTrsm(layout, side, triangle, a_transpose, diagonal,
                   m, n,
                   alpha,
                   Buffer<T>(a_buffer), a_offset, a_ld,
                   Buffer<T>(b_buffer), b_offset, b_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
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
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
template <typename T>
StatusCode Omatcopy(const Layout layout, const Transpose a_transpose,
                    const size_t m, const size_t n,
                    const T alpha,
                    const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                    cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                    cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xomatcopy<T>(queue_cpp, event);
    routine.DoOmatcopy(layout, a_transpose,
                       m, n,
                       alpha,
                       Buffer<T>(a_buffer), a_offset, a_ld,
                       Buffer<T>(b_buffer), b_offset, b_ld);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
}
template StatusCode PUBLIC_API Omatcopy<float>(const Layout, const Transpose,
                                               const size_t, const size_t,
                                               const float,
                                               const cl_mem, const size_t, const size_t,
                                               cl_mem, const size_t, const size_t,
                                               cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Omatcopy<double>(const Layout, const Transpose,
                                                const size_t, const size_t,
                                                const double,
                                                const cl_mem, const size_t, const size_t,
                                                cl_mem, const size_t, const size_t,
                                                cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Omatcopy<float2>(const Layout, const Transpose,
                                                const size_t, const size_t,
                                                const float2,
                                                const cl_mem, const size_t, const size_t,
                                                cl_mem, const size_t, const size_t,
                                                cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Omatcopy<double2>(const Layout, const Transpose,
                                                 const size_t, const size_t,
                                                 const double2,
                                                 const cl_mem, const size_t, const size_t,
                                                 cl_mem, const size_t, const size_t,
                                                 cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Omatcopy<half>(const Layout, const Transpose,
                                              const size_t, const size_t,
                                              const half,
                                              const cl_mem, const size_t, const size_t,
                                              cl_mem, const size_t, const size_t,
                                              cl_command_queue*, cl_event*);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
template <typename T>
StatusCode Im2col(const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                  const cl_mem im_buffer, const size_t im_offset,
                  cl_mem col_buffer, const size_t col_offset,
                  cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = Xim2col<T>(queue_cpp, event);
    routine.DoIm2col(channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                     Buffer<T>(im_buffer), im_offset,
                     Buffer<T>(col_buffer), col_offset);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
}
template StatusCode PUBLIC_API Im2col<float>(const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                             const cl_mem, const size_t,
                                             cl_mem, const size_t,
                                             cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Im2col<double>(const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                              const cl_mem, const size_t,
                                              cl_mem, const size_t,
                                              cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Im2col<float2>(const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                              const cl_mem, const size_t,
                                              cl_mem, const size_t,
                                              cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Im2col<double2>(const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                               const cl_mem, const size_t,
                                               cl_mem, const size_t,
                                               cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API Im2col<half>(const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                            const cl_mem, const size_t,
                                            cl_mem, const size_t,
                                            cl_command_queue*, cl_event*);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
template <typename T>
StatusCode AxpyBatched(const size_t n,
                       const T *alphas,
                       const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                       cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                       const size_t batch_count,
                       cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = XaxpyBatched<T>(queue_cpp, event);
    auto alphas_cpp = std::vector<T>();
    auto x_offsets_cpp = std::vector<size_t>();
    auto y_offsets_cpp = std::vector<size_t>();
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
      alphas_cpp.push_back(alphas[batch]);
      x_offsets_cpp.push_back(x_offsets[batch]);
      y_offsets_cpp.push_back(y_offsets[batch]);
    }
    routine.DoAxpyBatched(n,
                          alphas_cpp,
                          Buffer<T>(x_buffer), x_offsets_cpp, x_inc,
                          Buffer<T>(y_buffer), y_offsets_cpp, y_inc,
                          batch_count);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
}
template StatusCode PUBLIC_API AxpyBatched<float>(const size_t,
                                                  const float*,
                                                  const cl_mem, const size_t*, const size_t,
                                                  cl_mem, const size_t*, const size_t,
                                                  const size_t,
                                                  cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API AxpyBatched<double>(const size_t,
                                                   const double*,
                                                   const cl_mem, const size_t*, const size_t,
                                                   cl_mem, const size_t*, const size_t,
                                                   const size_t,
                                                   cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API AxpyBatched<float2>(const size_t,
                                                   const float2*,
                                                   const cl_mem, const size_t*, const size_t,
                                                   cl_mem, const size_t*, const size_t,
                                                   const size_t,
                                                   cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API AxpyBatched<double2>(const size_t,
                                                    const double2*,
                                                    const cl_mem, const size_t*, const size_t,
                                                    cl_mem, const size_t*, const size_t,
                                                    const size_t,
                                                    cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API AxpyBatched<half>(const size_t,
                                                 const half*,
                                                 const cl_mem, const size_t*, const size_t,
                                                 cl_mem, const size_t*, const size_t,
                                                 const size_t,
                                                 cl_command_queue*, cl_event*);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
template <typename T>
StatusCode GemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                       const size_t m, const size_t n, const size_t k,
                       const T *alphas,
                       const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                       const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                       const T *betas,
                       cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                       const size_t batch_count,
                       cl_command_queue* queue, cl_event* event) {
  try {
    auto queue_cpp = Queue(*queue);
    auto routine = XgemmBatched<T>(queue_cpp, event);
    auto alphas_cpp = std::vector<T>();
    auto betas_cpp = std::vector<T>();
    auto a_offsets_cpp = std::vector<size_t>();
    auto b_offsets_cpp = std::vector<size_t>();
    auto c_offsets_cpp = std::vector<size_t>();
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
      alphas_cpp.push_back(alphas[batch]);
      betas_cpp.push_back(betas[batch]);
      a_offsets_cpp.push_back(a_offsets[batch]);
      b_offsets_cpp.push_back(b_offsets[batch]);
      c_offsets_cpp.push_back(c_offsets[batch]);
    }
    routine.DoGemmBatched(layout, a_transpose, b_transpose,
                          m, n, k,
                          alphas_cpp,
                          Buffer<T>(a_buffer), a_offsets_cpp, a_ld,
                          Buffer<T>(b_buffer), b_offsets_cpp, b_ld,
                          betas_cpp,
                          Buffer<T>(c_buffer), c_offsets_cpp, c_ld,
                          batch_count);
    return StatusCode::kSuccess;
  } catch (...) { return DispatchException(); }
}
template StatusCode PUBLIC_API GemmBatched<float>(const Layout, const Transpose, const Transpose,
                                                  const size_t, const size_t, const size_t,
                                                  const float*,
                                                  const cl_mem, const size_t*, const size_t,
                                                  const cl_mem, const size_t*, const size_t,
                                                  const float*,
                                                  cl_mem, const size_t*, const size_t,
                                                  const size_t,
                                                  cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API GemmBatched<double>(const Layout, const Transpose, const Transpose,
                                                   const size_t, const size_t, const size_t,
                                                   const double*,
                                                   const cl_mem, const size_t*, const size_t,
                                                   const cl_mem, const size_t*, const size_t,
                                                   const double*,
                                                   cl_mem, const size_t*, const size_t,
                                                   const size_t,
                                                   cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API GemmBatched<float2>(const Layout, const Transpose, const Transpose,
                                                   const size_t, const size_t, const size_t,
                                                   const float2*,
                                                   const cl_mem, const size_t*, const size_t,
                                                   const cl_mem, const size_t*, const size_t,
                                                   const float2*,
                                                   cl_mem, const size_t*, const size_t,
                                                   const size_t,
                                                   cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API GemmBatched<double2>(const Layout, const Transpose, const Transpose,
                                                    const size_t, const size_t, const size_t,
                                                    const double2*,
                                                    const cl_mem, const size_t*, const size_t,
                                                    const cl_mem, const size_t*, const size_t,
                                                    const double2*,
                                                    cl_mem, const size_t*, const size_t,
                                                    const size_t,
                                                    cl_command_queue*, cl_event*);
template StatusCode PUBLIC_API GemmBatched<half>(const Layout, const Transpose, const Transpose,
                                                 const size_t, const size_t, const size_t,
                                                 const half*,
                                                 const cl_mem, const size_t*, const size_t,
                                                 const cl_mem, const size_t*, const size_t,
                                                 const half*,
                                                 cl_mem, const size_t*, const size_t,
                                                 const size_t,
                                                 cl_command_queue*, cl_event*);
// =================================================================================================

// Clears the cache of stored binaries
StatusCode ClearCache() {
  try {
    ProgramCache::Instance().Invalidate();
    BinaryCache::Instance().Invalidate();
  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

template <typename Real, typename Complex>
void FillCacheForPrecision(Queue &queue) {
  try {

    // Runs all the level 1 set-up functions
    Xswap<Real>(queue, nullptr); Xswap<Complex>(queue, nullptr);
    Xswap<Real>(queue, nullptr); Xswap<Complex>(queue, nullptr);
    Xscal<Real>(queue, nullptr); Xscal<Complex>(queue, nullptr);
    Xcopy<Real>(queue, nullptr); Xcopy<Complex>(queue, nullptr);
    Xaxpy<Real>(queue, nullptr); Xaxpy<Complex>(queue, nullptr);
    Xdot<Real>(queue, nullptr);
    Xdotu<Complex>(queue, nullptr);
    Xdotc<Complex>(queue, nullptr);
    Xnrm2<Real>(queue, nullptr); Xnrm2<Complex>(queue, nullptr);
    Xasum<Real>(queue, nullptr); Xasum<Complex>(queue, nullptr);
    Xsum<Real>(queue, nullptr); Xsum<Complex>(queue, nullptr);
    Xamax<Real>(queue, nullptr); Xamax<Complex>(queue, nullptr);
    Xmax<Real>(queue, nullptr); Xmax<Complex>(queue, nullptr);
    Xmin<Real>(queue, nullptr); Xmin<Complex>(queue, nullptr);

    // Runs all the level 2 set-up functions
    Xgemv<Real>(queue, nullptr); Xgemv<Complex>(queue, nullptr);
    Xgbmv<Real>(queue, nullptr); Xgbmv<Complex>(queue, nullptr);
    Xhemv<Complex>(queue, nullptr);
    Xhbmv<Complex>(queue, nullptr);
    Xhpmv<Complex>(queue, nullptr);
    Xsymv<Real>(queue, nullptr);
    Xsbmv<Real>(queue, nullptr);
    Xspmv<Real>(queue, nullptr);
    Xtrmv<Real>(queue, nullptr); Xtrmv<Complex>(queue, nullptr);
    Xtbmv<Real>(queue, nullptr); Xtbmv<Complex>(queue, nullptr);
    Xtpmv<Real>(queue, nullptr); Xtpmv<Complex>(queue, nullptr);
    Xger<Real>(queue, nullptr);
    Xgeru<Complex>(queue, nullptr);
    Xgerc<Complex>(queue, nullptr);
    Xher<Complex,Real>(queue, nullptr);
    Xhpr<Complex,Real>(queue, nullptr);
    Xher2<Complex>(queue, nullptr);
    Xhpr2<Complex>(queue, nullptr);
    Xsyr<Real>(queue, nullptr);
    Xspr<Real>(queue, nullptr);
    Xsyr2<Real>(queue, nullptr);
    Xspr2<Real>(queue, nullptr);

    // Runs all the level 3 set-up functions
    Xgemm<Real>(queue, nullptr); Xgemm<Complex>(queue, nullptr);
    Xsymm<Real>(queue, nullptr); Xsymm<Complex>(queue, nullptr);
    Xhemm<Complex>(queue, nullptr);
    Xsyrk<Real>(queue, nullptr); Xsyrk<Complex>(queue, nullptr);
    Xherk<Complex,Real>(queue, nullptr);
    Xsyr2k<Real>(queue, nullptr); Xsyr2k<Complex>(queue, nullptr);
    Xher2k<Complex,Real>(queue, nullptr);
    Xtrmm<Real>(queue, nullptr); Xtrmm<Complex>(queue, nullptr);

    // Runs all the non-BLAS set-up functions
    Xomatcopy<Real>(queue, nullptr); Xomatcopy<Complex>(queue, nullptr);

  } catch(const RuntimeErrorCode &e) {
    if (e.status() != StatusCode::kNoDoublePrecision &&
        e.status() != StatusCode::kNoHalfPrecision) {
      throw;
    }
  }
}

// Fills the cache with all binaries for a specific device
// TODO: Add half-precision FP16 set-up calls
StatusCode FillCache(const cl_device_id device) {
  try {

    // Creates a sample context and queue to match the normal routine calling conventions
    auto device_cpp = Device(device);
    auto context = Context(device_cpp);
    auto queue = Queue(context, device_cpp);

    FillCacheForPrecision<float, float2>(queue);
    FillCacheForPrecision<double, double2>(queue);

  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

// =================================================================================================

// Overrides the tuning parameters for this device-precision-kernel combination
StatusCode OverrideParameters(const cl_device_id device, const std::string &kernel_name,
                              const Precision precision,
                              const std::unordered_map<std::string,size_t> &parameters) {
  try {

    // Retrieves the device name
    const auto device_cpp = Device(device);
    const auto device_name = device_cpp.Name();

    // Retrieves the current database values to verify whether the new ones are complete
    auto in_cache = false;
    const auto current_database = DatabaseCache::Instance().Get(DatabaseKeyRef{ precision, device_name, kernel_name }, &in_cache);
    if (!in_cache) { return StatusCode::kInvalidOverrideKernel; }
    for (const auto &current_param : current_database.GetParameterNames()) {
      if (parameters.find(current_param) == parameters.end()) {
        return StatusCode::kMissingOverrideParameter;
      }
    }

    // Clears the existing program & binary cache for routines with the target kernel
    const auto routine_names = Routine::routines_by_kernel.at(kernel_name);
    for (const auto &routine_name : routine_names) {
      ProgramCache::Instance().RemoveBySubset<1, 2>(ProgramKey{nullptr, device, precision, routine_name});
      BinaryCache::Instance().Remove(BinaryKey{precision, routine_name, device_name});
    }

    // Retrieves the names and values separately
    auto parameter_values = std::vector<size_t>();
    auto parameter_names = std::vector<std::string>();
    for (const auto &parameter : parameters) {
      parameter_values.push_back(parameter.second);
      parameter_names.push_back(parameter.first);
    }

    // Creates a small custom database based on the provided parameters
    const auto database_device = database::DatabaseDevice{"default", parameter_values};
    const auto database_architecture = database::DatabaseArchitecture{"default", {database_device}};
    const auto database_vendor = database::DatabaseVendor{database::kDeviceTypeAll, "default", {database_architecture}};
    const auto database_entry = database::DatabaseEntry{kernel_name, precision, parameter_names, {database_vendor}};
    const auto database_entries = std::vector<database::DatabaseEntry>{database_entry};
    const auto database = Database(device_cpp, kernel_name, precision, database_entries);

    // Removes the old database entry and stores the new one in the cache
    DatabaseCache::Instance().Remove(DatabaseKey{ precision, device_name, kernel_name });
    DatabaseCache::Instance().Store(DatabaseKey{ precision, device_name, kernel_name }, Database(database));

  } catch (...) { return DispatchException(); }
  return StatusCode::kSuccess;
}

// =================================================================================================
} // namespace clblast
