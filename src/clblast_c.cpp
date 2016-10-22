
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

#include "clblast_c.h"
#include "clblast.h"
#include "utilities/utilities.hpp"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// ROTG
StatusCode CLBlastSrotg(cl_mem sa_buffer, const size_t sa_offset,
                        cl_mem sb_buffer, const size_t sb_offset,
                        cl_mem sc_buffer, const size_t sc_offset,
                        cl_mem ss_buffer, const size_t ss_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rotg<float>(sa_buffer, sa_offset,
                                                        sb_buffer, sb_offset,
                                                        sc_buffer, sc_offset,
                                                        ss_buffer, ss_offset,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDrotg(cl_mem sa_buffer, const size_t sa_offset,
                        cl_mem sb_buffer, const size_t sb_offset,
                        cl_mem sc_buffer, const size_t sc_offset,
                        cl_mem ss_buffer, const size_t ss_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rotg<double>(sa_buffer, sa_offset,
                                                         sb_buffer, sb_offset,
                                                         sc_buffer, sc_offset,
                                                         ss_buffer, ss_offset,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// ROTMG
StatusCode CLBlastSrotmg(cl_mem sd1_buffer, const size_t sd1_offset,
                         cl_mem sd2_buffer, const size_t sd2_offset,
                         cl_mem sx1_buffer, const size_t sx1_offset,
                         const cl_mem sy1_buffer, const size_t sy1_offset,
                         cl_mem sparam_buffer, const size_t sparam_offset,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rotmg<float>(sd1_buffer, sd1_offset,
                                                         sd2_buffer, sd2_offset,
                                                         sx1_buffer, sx1_offset,
                                                         sy1_buffer, sy1_offset,
                                                         sparam_buffer, sparam_offset,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDrotmg(cl_mem sd1_buffer, const size_t sd1_offset,
                         cl_mem sd2_buffer, const size_t sd2_offset,
                         cl_mem sx1_buffer, const size_t sx1_offset,
                         const cl_mem sy1_buffer, const size_t sy1_offset,
                         cl_mem sparam_buffer, const size_t sparam_offset,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rotmg<double>(sd1_buffer, sd1_offset,
                                                          sd2_buffer, sd2_offset,
                                                          sx1_buffer, sx1_offset,
                                                          sy1_buffer, sy1_offset,
                                                          sparam_buffer, sparam_offset,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// ROT
StatusCode CLBlastSrot(const size_t n,
                       cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       const float cos,
                       const float sin,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rot(n,
                                                x_buffer, x_offset, x_inc,
                                                y_buffer, y_offset, y_inc,
                                                cos,
                                                sin,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDrot(const size_t n,
                       cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       const double cos,
                       const double sin,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rot(n,
                                                x_buffer, x_offset, x_inc,
                                                y_buffer, y_offset, y_inc,
                                                cos,
                                                sin,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// ROTM
StatusCode CLBlastSrotm(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem sparam_buffer, const size_t sparam_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rotm<float>(n,
                                                        x_buffer, x_offset, x_inc,
                                                        y_buffer, y_offset, y_inc,
                                                        sparam_buffer, sparam_offset,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDrotm(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem sparam_buffer, const size_t sparam_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Rotm<double>(n,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         sparam_buffer, sparam_offset,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SWAP
StatusCode CLBlastSswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Swap<float>(n,
                                                        x_buffer, x_offset, x_inc,
                                                        y_buffer, y_offset, y_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Swap<double>(n,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Swap<float2>(n,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Swap<double2>(n,
                                                          x_buffer, x_offset, x_inc,
                                                          y_buffer, y_offset, y_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHswap(const size_t n,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Swap<half>(n,
                                                       x_buffer, x_offset, x_inc,
                                                       y_buffer, y_offset, y_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SCAL
StatusCode CLBlastSscal(const size_t n,
                        const float alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Scal(n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDscal(const size_t n,
                        const double alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Scal(n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCscal(const size_t n,
                        const cl_float2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Scal(n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZscal(const size_t n,
                        const cl_double2 alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Scal(n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHscal(const size_t n,
                        const cl_half alpha,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Scal(n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// COPY
StatusCode CLBlastScopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Copy<float>(n,
                                                        x_buffer, x_offset, x_inc,
                                                        y_buffer, y_offset, y_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Copy<double>(n,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Copy<float2>(n,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Copy<double2>(n,
                                                          x_buffer, x_offset, x_inc,
                                                          y_buffer, y_offset, y_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHcopy(const size_t n,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Copy<half>(n,
                                                       x_buffer, x_offset, x_inc,
                                                       y_buffer, y_offset, y_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// AXPY
StatusCode CLBlastSaxpy(const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Axpy(n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDaxpy(const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Axpy(n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCaxpy(const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Axpy(n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZaxpy(const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Axpy(n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHaxpy(const size_t n,
                        const cl_half alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Axpy(n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// DOT
StatusCode CLBlastSdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dot<float>(n,
                                                       dot_buffer, dot_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       y_buffer, y_offset, y_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dot<double>(n,
                                                        dot_buffer, dot_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        y_buffer, y_offset, y_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHdot(const size_t n,
                       cl_mem dot_buffer, const size_t dot_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dot<half>(n,
                                                      dot_buffer, dot_offset,
                                                      x_buffer, x_offset, x_inc,
                                                      y_buffer, y_offset, y_inc,
                                                      queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// DOTU
StatusCode CLBlastCdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dotu<float2>(n,
                                                         dot_buffer, dot_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZdotu(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dotu<double2>(n,
                                                          dot_buffer, dot_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          y_buffer, y_offset, y_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// DOTC
StatusCode CLBlastCdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dotc<float2>(n,
                                                         dot_buffer, dot_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         y_buffer, y_offset, y_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZdotc(const size_t n,
                        cl_mem dot_buffer, const size_t dot_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Dotc<double2>(n,
                                                          dot_buffer, dot_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          y_buffer, y_offset, y_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// NRM2
StatusCode CLBlastSnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Nrm2<float>(n,
                                                        nrm2_buffer, nrm2_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Nrm2<double>(n,
                                                         nrm2_buffer, nrm2_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastScnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Nrm2<float2>(n,
                                                         nrm2_buffer, nrm2_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDznrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Nrm2<double2>(n,
                                                          nrm2_buffer, nrm2_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHnrm2(const size_t n,
                        cl_mem nrm2_buffer, const size_t nrm2_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Nrm2<half>(n,
                                                       nrm2_buffer, nrm2_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// ASUM
StatusCode CLBlastSasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Asum<float>(n,
                                                        asum_buffer, asum_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Asum<double>(n,
                                                         asum_buffer, asum_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastScasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Asum<float2>(n,
                                                         asum_buffer, asum_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDzasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Asum<double2>(n,
                                                          asum_buffer, asum_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHasum(const size_t n,
                        cl_mem asum_buffer, const size_t asum_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Asum<half>(n,
                                                       asum_buffer, asum_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SUM
StatusCode CLBlastSsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sum<float>(n,
                                                       sum_buffer, sum_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sum<double>(n,
                                                        sum_buffer, sum_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastScsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sum<float2>(n,
                                                        sum_buffer, sum_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDzsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sum<double2>(n,
                                                         sum_buffer, sum_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsum(const size_t n,
                       cl_mem sum_buffer, const size_t sum_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sum<half>(n,
                                                      sum_buffer, sum_offset,
                                                      x_buffer, x_offset, x_inc,
                                                      queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// AMAX
StatusCode CLBlastiSamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Amax<float>(n,
                                                        imax_buffer, imax_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiDamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Amax<double>(n,
                                                         imax_buffer, imax_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiCamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Amax<float2>(n,
                                                         imax_buffer, imax_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiZamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Amax<double2>(n,
                                                          imax_buffer, imax_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiHamax(const size_t n,
                        cl_mem imax_buffer, const size_t imax_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Amax<half>(n,
                                                       imax_buffer, imax_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// MAX
StatusCode CLBlastiSmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Max<float>(n,
                                                       imax_buffer, imax_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiDmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Max<double>(n,
                                                        imax_buffer, imax_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiCmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Max<float2>(n,
                                                        imax_buffer, imax_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiZmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Max<double2>(n,
                                                         imax_buffer, imax_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiHmax(const size_t n,
                       cl_mem imax_buffer, const size_t imax_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Max<half>(n,
                                                      imax_buffer, imax_offset,
                                                      x_buffer, x_offset, x_inc,
                                                      queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// MIN
StatusCode CLBlastiSmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Min<float>(n,
                                                       imin_buffer, imin_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiDmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Min<double>(n,
                                                        imin_buffer, imin_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiCmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Min<float2>(n,
                                                        imin_buffer, imin_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiZmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Min<double2>(n,
                                                         imin_buffer, imin_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastiHmin(const size_t n,
                       cl_mem imin_buffer, const size_t imin_offset,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Min<half>(n,
                                                      imin_buffer, imin_offset,
                                                      x_buffer, x_offset, x_inc,
                                                      queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Gemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 float2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 double2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHgemv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_half beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Gbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n, kl, ku,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n, kl, ku,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_float2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n, kl, ku,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 float2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n, kl, ku,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 double2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHgbmv(const Layout layout, const Transpose a_transpose,
                        const size_t m, const size_t n, const size_t kl, const size_t ku,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_half beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 m, n, kl, ku,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Hemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 float2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZhemv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hemv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 double2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Hbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n, k,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 float2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZhbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n, k,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 double2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Hpmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 ap_buffer, ap_offset,
                                                 x_buffer, x_offset, x_inc,
                                                 float2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZhpmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_double2 beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hpmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 ap_buffer, ap_offset,
                                                 x_buffer, x_offset, x_inc,
                                                 double2{beta.s[0], beta.s[1]},
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Symv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Symv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsymv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_half beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Symv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Sbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsbmv(const Layout layout, const Triangle triangle,
                        const size_t n, const size_t k,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_half beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Sbmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Spmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 ap_buffer, ap_offset,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDspmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const double beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 ap_buffer, ap_offset,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHspmv(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_half alpha,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_half beta,
                        cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spmv(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 ap_buffer, ap_offset,
                                                 x_buffer, x_offset, x_inc,
                                                 beta,
                                                 y_buffer, y_offset, y_inc,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TRMV
StatusCode CLBlastStrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmv<float>(static_cast<clblast::Layout>(layout),
                                                        static_cast<clblast::Triangle>(triangle),
                                                        static_cast<clblast::Transpose>(a_transpose),
                                                        static_cast<clblast::Diagonal>(diagonal),
                                                        n,
                                                        a_buffer, a_offset, a_ld,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmv<double>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmv<float2>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmv<double2>(static_cast<clblast::Layout>(layout),
                                                          static_cast<clblast::Triangle>(triangle),
                                                          static_cast<clblast::Transpose>(a_transpose),
                                                          static_cast<clblast::Diagonal>(diagonal),
                                                          n,
                                                          a_buffer, a_offset, a_ld,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHtrmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmv<half>(static_cast<clblast::Layout>(layout),
                                                       static_cast<clblast::Triangle>(triangle),
                                                       static_cast<clblast::Transpose>(a_transpose),
                                                       static_cast<clblast::Diagonal>(diagonal),
                                                       n,
                                                       a_buffer, a_offset, a_ld,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TBMV
StatusCode CLBlastStbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbmv<float>(static_cast<clblast::Layout>(layout),
                                                        static_cast<clblast::Triangle>(triangle),
                                                        static_cast<clblast::Transpose>(a_transpose),
                                                        static_cast<clblast::Diagonal>(diagonal),
                                                        n, k,
                                                        a_buffer, a_offset, a_ld,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbmv<double>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n, k,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbmv<float2>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n, k,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbmv<double2>(static_cast<clblast::Layout>(layout),
                                                          static_cast<clblast::Triangle>(triangle),
                                                          static_cast<clblast::Transpose>(a_transpose),
                                                          static_cast<clblast::Diagonal>(diagonal),
                                                          n, k,
                                                          a_buffer, a_offset, a_ld,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHtbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbmv<half>(static_cast<clblast::Layout>(layout),
                                                       static_cast<clblast::Triangle>(triangle),
                                                       static_cast<clblast::Transpose>(a_transpose),
                                                       static_cast<clblast::Diagonal>(diagonal),
                                                       n, k,
                                                       a_buffer, a_offset, a_ld,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TPMV
StatusCode CLBlastStpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpmv<float>(static_cast<clblast::Layout>(layout),
                                                        static_cast<clblast::Triangle>(triangle),
                                                        static_cast<clblast::Transpose>(a_transpose),
                                                        static_cast<clblast::Diagonal>(diagonal),
                                                        n,
                                                        ap_buffer, ap_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpmv<double>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         ap_buffer, ap_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpmv<float2>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         ap_buffer, ap_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpmv<double2>(static_cast<clblast::Layout>(layout),
                                                          static_cast<clblast::Triangle>(triangle),
                                                          static_cast<clblast::Transpose>(a_transpose),
                                                          static_cast<clblast::Diagonal>(diagonal),
                                                          n,
                                                          ap_buffer, ap_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHtpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpmv<half>(static_cast<clblast::Layout>(layout),
                                                       static_cast<clblast::Triangle>(triangle),
                                                       static_cast<clblast::Transpose>(a_transpose),
                                                       static_cast<clblast::Diagonal>(diagonal),
                                                       n,
                                                       ap_buffer, ap_offset,
                                                       x_buffer, x_offset, x_inc,
                                                       queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TRSV
StatusCode CLBlastStrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsv<float>(static_cast<clblast::Layout>(layout),
                                                        static_cast<clblast::Triangle>(triangle),
                                                        static_cast<clblast::Transpose>(a_transpose),
                                                        static_cast<clblast::Diagonal>(diagonal),
                                                        n,
                                                        a_buffer, a_offset, a_ld,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsv<double>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsv<float2>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsv<double2>(static_cast<clblast::Layout>(layout),
                                                          static_cast<clblast::Triangle>(triangle),
                                                          static_cast<clblast::Transpose>(a_transpose),
                                                          static_cast<clblast::Diagonal>(diagonal),
                                                          n,
                                                          a_buffer, a_offset, a_ld,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TBSV
StatusCode CLBlastStbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbsv<float>(static_cast<clblast::Layout>(layout),
                                                        static_cast<clblast::Triangle>(triangle),
                                                        static_cast<clblast::Transpose>(a_transpose),
                                                        static_cast<clblast::Diagonal>(diagonal),
                                                        n, k,
                                                        a_buffer, a_offset, a_ld,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbsv<double>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n, k,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbsv<float2>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n, k,
                                                         a_buffer, a_offset, a_ld,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n, const size_t k,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tbsv<double2>(static_cast<clblast::Layout>(layout),
                                                          static_cast<clblast::Triangle>(triangle),
                                                          static_cast<clblast::Transpose>(a_transpose),
                                                          static_cast<clblast::Diagonal>(diagonal),
                                                          n, k,
                                                          a_buffer, a_offset, a_ld,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TPSV
StatusCode CLBlastStpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpsv<float>(static_cast<clblast::Layout>(layout),
                                                        static_cast<clblast::Triangle>(triangle),
                                                        static_cast<clblast::Transpose>(a_transpose),
                                                        static_cast<clblast::Diagonal>(diagonal),
                                                        n,
                                                        ap_buffer, ap_offset,
                                                        x_buffer, x_offset, x_inc,
                                                        queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpsv<double>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         ap_buffer, ap_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpsv<float2>(static_cast<clblast::Layout>(layout),
                                                         static_cast<clblast::Triangle>(triangle),
                                                         static_cast<clblast::Transpose>(a_transpose),
                                                         static_cast<clblast::Diagonal>(diagonal),
                                                         n,
                                                         ap_buffer, ap_offset,
                                                         x_buffer, x_offset, x_inc,
                                                         queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t n,
                        const cl_mem ap_buffer, const size_t ap_offset,
                        cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Tpsv<double2>(static_cast<clblast::Layout>(layout),
                                                          static_cast<clblast::Triangle>(triangle),
                                                          static_cast<clblast::Transpose>(a_transpose),
                                                          static_cast<clblast::Diagonal>(diagonal),
                                                          n,
                                                          ap_buffer, ap_offset,
                                                          x_buffer, x_offset, x_inc,
                                                          queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// GER
StatusCode CLBlastSger(const Layout layout,
                       const size_t m, const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Ger(static_cast<clblast::Layout>(layout),
                                                m, n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                y_buffer, y_offset, y_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDger(const Layout layout,
                       const size_t m, const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Ger(static_cast<clblast::Layout>(layout),
                                                m, n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                y_buffer, y_offset, y_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHger(const Layout layout,
                       const size_t m, const size_t n,
                       const cl_half alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Ger(static_cast<clblast::Layout>(layout),
                                                m, n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                y_buffer, y_offset, y_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// GERU
StatusCode CLBlastCgeru(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Geru(static_cast<clblast::Layout>(layout),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZgeru(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Geru(static_cast<clblast::Layout>(layout),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// GERC
StatusCode CLBlastCgerc(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gerc(static_cast<clblast::Layout>(layout),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZgerc(const Layout layout,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gerc(static_cast<clblast::Layout>(layout),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// HER
StatusCode CLBlastCher(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Her(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZher(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Her(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// HPR
StatusCode CLBlastChpr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hpr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                ap_buffer, ap_offset,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZhpr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hpr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                ap_buffer, ap_offset,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// HER2
StatusCode CLBlastCher2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Her2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZher2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Her2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// HPR2
StatusCode CLBlastChpr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_float2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hpr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 ap_buffer, ap_offset,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZhpr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_double2 alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hpr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 ap_buffer, ap_offset,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SYR
StatusCode CLBlastSsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsyr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const cl_half alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                a_buffer, a_offset, a_ld,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SPR
StatusCode CLBlastSspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const float alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                ap_buffer, ap_offset,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const double alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                ap_buffer, ap_offset,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHspr(const Layout layout, const Triangle triangle,
                       const size_t n,
                       const cl_half alpha,
                       const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                       cl_mem ap_buffer, const size_t ap_offset,
                       cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spr(static_cast<clblast::Layout>(layout),
                                                static_cast<clblast::Triangle>(triangle),
                                                n,
                                                alpha,
                                                x_buffer, x_offset, x_inc,
                                                ap_buffer, ap_offset,
                                                queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SYR2
StatusCode CLBlastSsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsyr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_half alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 a_buffer, a_offset, a_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SPR2
StatusCode CLBlastSspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const float alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 ap_buffer, ap_offset,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const double alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 ap_buffer, ap_offset,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHspr2(const Layout layout, const Triangle triangle,
                        const size_t n,
                        const cl_half alpha,
                        const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                        const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                        cl_mem ap_buffer, const size_t ap_offset,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Spr2(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 n,
                                                 alpha,
                                                 x_buffer, x_offset, x_inc,
                                                 y_buffer, y_offset, y_inc,
                                                 ap_buffer, ap_offset,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Gemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Transpose>(b_transpose),
                                                 m, n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Transpose>(b_transpose),
                                                 m, n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Transpose>(b_transpose),
                                                 m, n, k,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 float2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Transpose>(b_transpose),
                                                 m, n, k,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 double2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHgemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_half beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Gemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Transpose>(b_transpose),
                                                 m, n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Symm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Symm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Symm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 float2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Symm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 double2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsymm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_half beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Symm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Hemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 float2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZhemm(const Layout layout, const Side side, const Triangle triangle,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Hemm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 double2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// SYRK
StatusCode CLBlastSsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syrk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syrk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_float2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syrk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 float2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_double2 beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syrk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 double2{beta.s[0], beta.s[1]},
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const cl_half beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syrk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// HERK
StatusCode CLBlastCherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const float beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Herk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZherk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                        const size_t n, const size_t k,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        const double beta,
                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Herk(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 n, k,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 beta,
                                                 c_buffer, c_offset, c_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Syr2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  alpha,
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  beta,
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const double alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  alpha,
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  beta,
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_float2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_float2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  float2{alpha.s[0], alpha.s[1]},
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  float2{beta.s[0], beta.s[1]},
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_double2 beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  double2{alpha.s[0], alpha.s[1]},
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  double2{beta.s[0], beta.s[1]},
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHsyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_half alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const cl_half beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Syr2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  alpha,
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  beta,
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
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
  try {
    return static_cast<StatusCode>(clblast::Her2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  float2{alpha.s[0], alpha.s[1]},
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  beta,
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZher2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                         const size_t n, const size_t k,
                         const cl_double2 alpha,
                         const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                         const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                         const double beta,
                         cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                         cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Her2k(static_cast<clblast::Layout>(layout),
                                                  static_cast<clblast::Triangle>(triangle),
                                                  static_cast<clblast::Transpose>(ab_transpose),
                                                  n, k,
                                                  double2{alpha.s[0], alpha.s[1]},
                                                  a_buffer, a_offset, a_ld,
                                                  b_buffer, b_offset, b_ld,
                                                  beta,
                                                  c_buffer, c_offset, c_ld,
                                                  queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TRMM
StatusCode CLBlastStrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHtrmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trmm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// TRSM
StatusCode CLBlastStrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const float alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const double alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastCtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_float2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 float2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_double2 alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 double2{alpha.s[0], alpha.s[1]},
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHtrsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                        const size_t m, const size_t n,
                        const cl_half alpha,
                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                        cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                        cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Trsm(static_cast<clblast::Layout>(layout),
                                                 static_cast<clblast::Side>(side),
                                                 static_cast<clblast::Triangle>(triangle),
                                                 static_cast<clblast::Transpose>(a_transpose),
                                                 static_cast<clblast::Diagonal>(diagonal),
                                                 m, n,
                                                 alpha,
                                                 a_buffer, a_offset, a_ld,
                                                 b_buffer, b_offset, b_ld,
                                                 queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// OMATCOPY
StatusCode CLBlastSomatcopy(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n,
                            const float alpha,
                            const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                            cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                            cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                                                     static_cast<clblast::Transpose>(a_transpose),
                                                     m, n,
                                                     alpha,
                                                     a_buffer, a_offset, a_ld,
                                                     b_buffer, b_offset, b_ld,
                                                     queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastDomatcopy(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n,
                            const double alpha,
                            const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                            cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                            cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                                                     static_cast<clblast::Transpose>(a_transpose),
                                                     m, n,
                                                     alpha,
                                                     a_buffer, a_offset, a_ld,
                                                     b_buffer, b_offset, b_ld,
                                                     queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastComatcopy(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n,
                            const cl_float2 alpha,
                            const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                            cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                            cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                                                     static_cast<clblast::Transpose>(a_transpose),
                                                     m, n,
                                                     float2{alpha.s[0], alpha.s[1]},
                                                     a_buffer, a_offset, a_ld,
                                                     b_buffer, b_offset, b_ld,
                                                     queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastZomatcopy(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n,
                            const cl_double2 alpha,
                            const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                            cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                            cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                                                     static_cast<clblast::Transpose>(a_transpose),
                                                     m, n,
                                                     double2{alpha.s[0], alpha.s[1]},
                                                     a_buffer, a_offset, a_ld,
                                                     b_buffer, b_offset, b_ld,
                                                     queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}
StatusCode CLBlastHomatcopy(const Layout layout, const Transpose a_transpose,
                            const size_t m, const size_t n,
                            const cl_half alpha,
                            const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                            cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                            cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<StatusCode>(clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                                                     static_cast<clblast::Transpose>(a_transpose),
                                                     m, n,
                                                     alpha,
                                                     a_buffer, a_offset, a_ld,
                                                     b_buffer, b_offset, b_ld,
                                                     queue, event));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================

// Clears the cache of stored binaries
StatusCode CLBlastClearCache() {
  try {
    return static_cast<StatusCode>(clblast::ClearCache());
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// Fills the cache with binaries for a specific device
StatusCode CLBlastFillCache(const cl_device_id device) {
  try {
    return static_cast<StatusCode>(clblast::FillCache(device));
  } catch (...) { return static_cast<StatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================
