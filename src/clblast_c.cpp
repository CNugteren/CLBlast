
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
#include <unordered_map>

#include "utilities/utilities.hpp"
#include "clblast_c.h"
#include "clblast.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// ROTG
CLBlastStatusCode CLBlastSrotg(cl_mem sa_buffer, const size_t sa_offset,
                               cl_mem sb_buffer, const size_t sb_offset,
                               cl_mem sc_buffer, const size_t sc_offset,
                               cl_mem ss_buffer, const size_t ss_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rotg<float>(sa_buffer, sa_offset,
                           sb_buffer, sb_offset,
                           sc_buffer, sc_offset,
                           ss_buffer, ss_offset,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDrotg(cl_mem sa_buffer, const size_t sa_offset,
                               cl_mem sb_buffer, const size_t sb_offset,
                               cl_mem sc_buffer, const size_t sc_offset,
                               cl_mem ss_buffer, const size_t ss_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rotg<double>(sa_buffer, sa_offset,
                            sb_buffer, sb_offset,
                            sc_buffer, sc_offset,
                            ss_buffer, ss_offset,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// ROTMG
CLBlastStatusCode CLBlastSrotmg(cl_mem sd1_buffer, const size_t sd1_offset,
                                cl_mem sd2_buffer, const size_t sd2_offset,
                                cl_mem sx1_buffer, const size_t sx1_offset,
                                const cl_mem sy1_buffer, const size_t sy1_offset,
                                cl_mem sparam_buffer, const size_t sparam_offset,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rotmg<float>(sd1_buffer, sd1_offset,
                            sd2_buffer, sd2_offset,
                            sx1_buffer, sx1_offset,
                            sy1_buffer, sy1_offset,
                            sparam_buffer, sparam_offset,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDrotmg(cl_mem sd1_buffer, const size_t sd1_offset,
                                cl_mem sd2_buffer, const size_t sd2_offset,
                                cl_mem sx1_buffer, const size_t sx1_offset,
                                const cl_mem sy1_buffer, const size_t sy1_offset,
                                cl_mem sparam_buffer, const size_t sparam_offset,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rotmg<double>(sd1_buffer, sd1_offset,
                             sd2_buffer, sd2_offset,
                             sx1_buffer, sx1_offset,
                             sy1_buffer, sy1_offset,
                             sparam_buffer, sparam_offset,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// ROT
CLBlastStatusCode CLBlastSrot(const size_t n,
                              cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const float cos,
                              const float sin,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rot(n,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   cos,
                   sin,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDrot(const size_t n,
                              cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              const double cos,
                              const double sin,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rot(n,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   cos,
                   sin,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// ROTM
CLBlastStatusCode CLBlastSrotm(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem sparam_buffer, const size_t sparam_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rotm<float>(n,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc,
                           sparam_buffer, sparam_offset,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDrotm(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem sparam_buffer, const size_t sparam_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Rotm<double>(n,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            sparam_buffer, sparam_offset,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SWAP
CLBlastStatusCode CLBlastSswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Swap<float>(n,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Swap<double>(n,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Swap<float2>(n,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Swap<double2>(n,
                             x_buffer, x_offset, x_inc,
                             y_buffer, y_offset, y_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHswap(const size_t n,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Swap<half>(n,
                          x_buffer, x_offset, x_inc,
                          y_buffer, y_offset, y_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SCAL
CLBlastStatusCode CLBlastSscal(const size_t n,
                               const float alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Scal(n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDscal(const size_t n,
                               const double alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Scal(n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCscal(const size_t n,
                               const cl_float2 alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Scal(n,
                    float2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZscal(const size_t n,
                               const cl_double2 alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Scal(n,
                    double2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHscal(const size_t n,
                               const cl_half alpha,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Scal(n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// COPY
CLBlastStatusCode CLBlastScopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Copy<float>(n,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Copy<double>(n,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Copy<float2>(n,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Copy<double2>(n,
                             x_buffer, x_offset, x_inc,
                             y_buffer, y_offset, y_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHcopy(const size_t n,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Copy<half>(n,
                          x_buffer, x_offset, x_inc,
                          y_buffer, y_offset, y_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// AXPY
CLBlastStatusCode CLBlastSaxpy(const size_t n,
                               const float alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Axpy(n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDaxpy(const size_t n,
                               const double alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Axpy(n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCaxpy(const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Axpy(n,
                    float2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZaxpy(const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Axpy(n,
                    double2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHaxpy(const size_t n,
                               const cl_half alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Axpy(n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// DOT
CLBlastStatusCode CLBlastSdot(const size_t n,
                              cl_mem dot_buffer, const size_t dot_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dot<float>(n,
                          dot_buffer, dot_offset,
                          x_buffer, x_offset, x_inc,
                          y_buffer, y_offset, y_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDdot(const size_t n,
                              cl_mem dot_buffer, const size_t dot_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dot<double>(n,
                           dot_buffer, dot_offset,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHdot(const size_t n,
                              cl_mem dot_buffer, const size_t dot_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dot<half>(n,
                         dot_buffer, dot_offset,
                         x_buffer, x_offset, x_inc,
                         y_buffer, y_offset, y_inc,
                         queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// DOTU
CLBlastStatusCode CLBlastCdotu(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dotu<float2>(n,
                            dot_buffer, dot_offset,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZdotu(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dotu<double2>(n,
                             dot_buffer, dot_offset,
                             x_buffer, x_offset, x_inc,
                             y_buffer, y_offset, y_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// DOTC
CLBlastStatusCode CLBlastCdotc(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dotc<float2>(n,
                            dot_buffer, dot_offset,
                            x_buffer, x_offset, x_inc,
                            y_buffer, y_offset, y_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZdotc(const size_t n,
                               cl_mem dot_buffer, const size_t dot_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Dotc<double2>(n,
                             dot_buffer, dot_offset,
                             x_buffer, x_offset, x_inc,
                             y_buffer, y_offset, y_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// NRM2
CLBlastStatusCode CLBlastSnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Nrm2<float>(n,
                           nrm2_buffer, nrm2_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Nrm2<double>(n,
                            nrm2_buffer, nrm2_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastScnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Nrm2<float2>(n,
                            nrm2_buffer, nrm2_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDznrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Nrm2<double2>(n,
                             nrm2_buffer, nrm2_offset,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHnrm2(const size_t n,
                               cl_mem nrm2_buffer, const size_t nrm2_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Nrm2<half>(n,
                          nrm2_buffer, nrm2_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// ASUM
CLBlastStatusCode CLBlastSasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Asum<float>(n,
                           asum_buffer, asum_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Asum<double>(n,
                            asum_buffer, asum_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastScasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Asum<float2>(n,
                            asum_buffer, asum_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDzasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Asum<double2>(n,
                             asum_buffer, asum_offset,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHasum(const size_t n,
                               cl_mem asum_buffer, const size_t asum_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Asum<half>(n,
                          asum_buffer, asum_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SUM
CLBlastStatusCode CLBlastSsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sum<float>(n,
                          sum_buffer, sum_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sum<double>(n,
                           sum_buffer, sum_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastScsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sum<float2>(n,
                           sum_buffer, sum_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDzsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sum<double2>(n,
                            sum_buffer, sum_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsum(const size_t n,
                              cl_mem sum_buffer, const size_t sum_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sum<half>(n,
                         sum_buffer, sum_offset,
                         x_buffer, x_offset, x_inc,
                         queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// AMAX
CLBlastStatusCode CLBlastiSamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Amax<float>(n,
                           imax_buffer, imax_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiDamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Amax<double>(n,
                            imax_buffer, imax_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiCamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Amax<float2>(n,
                            imax_buffer, imax_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiZamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Amax<double2>(n,
                             imax_buffer, imax_offset,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiHamax(const size_t n,
                               cl_mem imax_buffer, const size_t imax_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Amax<half>(n,
                          imax_buffer, imax_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// MAX
CLBlastStatusCode CLBlastiSmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Max<float>(n,
                          imax_buffer, imax_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiDmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Max<double>(n,
                           imax_buffer, imax_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiCmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Max<float2>(n,
                           imax_buffer, imax_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiZmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Max<double2>(n,
                            imax_buffer, imax_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiHmax(const size_t n,
                              cl_mem imax_buffer, const size_t imax_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Max<half>(n,
                         imax_buffer, imax_offset,
                         x_buffer, x_offset, x_inc,
                         queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// MIN
CLBlastStatusCode CLBlastiSmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Min<float>(n,
                          imin_buffer, imin_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiDmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Min<double>(n,
                           imin_buffer, imin_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiCmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Min<float2>(n,
                           imin_buffer, imin_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiZmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Min<double2>(n,
                            imin_buffer, imin_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastiHmin(const size_t n,
                              cl_mem imin_buffer, const size_t imin_offset,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Min<half>(n,
                         imin_buffer, imin_offset,
                         x_buffer, x_offset, x_inc,
                         queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// GEMV
CLBlastStatusCode CLBlastSgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    float2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    double2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// GBMV
CLBlastStatusCode CLBlastSgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n, kl, ku,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n, kl, ku,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n, kl, ku,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    float2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n, kl, ku,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    double2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                               const size_t m, const size_t n, const size_t kl, const size_t ku,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    m, n, kl, ku,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HEMV
CLBlastStatusCode CLBlastChemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    float2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZhemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hemv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    double2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HBMV
CLBlastStatusCode CLBlastChbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n, k,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    float2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZhbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n, k,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    double2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HPMV
CLBlastStatusCode CLBlastChpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_float2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hpmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    float2{alpha.s[0], alpha.s[1]},
                    ap_buffer, ap_offset,
                    x_buffer, x_offset, x_inc,
                    float2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZhpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_double2 beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hpmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    double2{alpha.s[0], alpha.s[1]},
                    ap_buffer, ap_offset,
                    x_buffer, x_offset, x_inc,
                    double2{beta.s[0], beta.s[1]},
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SYMV
CLBlastStatusCode CLBlastSsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SBMV
CLBlastStatusCode CLBlastSsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n, const size_t k,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Sbmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SPMV
CLBlastStatusCode CLBlastSspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const float beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    ap_buffer, ap_offset,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const double beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    ap_buffer, ap_offset,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_half beta,
                               cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spmv(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    ap_buffer, ap_offset,
                    x_buffer, x_offset, x_inc,
                    beta,
                    y_buffer, y_offset, y_inc,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TRMV
CLBlastStatusCode CLBlastStrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmv<float>(static_cast<clblast::Layout>(layout),
                           static_cast<clblast::Triangle>(triangle),
                           static_cast<clblast::Transpose>(a_transpose),
                           static_cast<clblast::Diagonal>(diagonal),
                           n,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmv<double>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmv<float2>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmv<double2>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             static_cast<clblast::Transpose>(a_transpose),
                             static_cast<clblast::Diagonal>(diagonal),
                             n,
                             a_buffer, a_offset, a_ld,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmv<half>(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(a_transpose),
                          static_cast<clblast::Diagonal>(diagonal),
                          n,
                          a_buffer, a_offset, a_ld,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TBMV
CLBlastStatusCode CLBlastStbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbmv<float>(static_cast<clblast::Layout>(layout),
                           static_cast<clblast::Triangle>(triangle),
                           static_cast<clblast::Transpose>(a_transpose),
                           static_cast<clblast::Diagonal>(diagonal),
                           n, k,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbmv<double>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n, k,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbmv<float2>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n, k,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbmv<double2>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             static_cast<clblast::Transpose>(a_transpose),
                             static_cast<clblast::Diagonal>(diagonal),
                             n, k,
                             a_buffer, a_offset, a_ld,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbmv<half>(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(a_transpose),
                          static_cast<clblast::Diagonal>(diagonal),
                          n, k,
                          a_buffer, a_offset, a_ld,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TPMV
CLBlastStatusCode CLBlastStpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpmv<float>(static_cast<clblast::Layout>(layout),
                           static_cast<clblast::Triangle>(triangle),
                           static_cast<clblast::Transpose>(a_transpose),
                           static_cast<clblast::Diagonal>(diagonal),
                           n,
                           ap_buffer, ap_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpmv<double>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            ap_buffer, ap_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpmv<float2>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            ap_buffer, ap_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpmv<double2>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             static_cast<clblast::Transpose>(a_transpose),
                             static_cast<clblast::Diagonal>(diagonal),
                             n,
                             ap_buffer, ap_offset,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpmv<half>(static_cast<clblast::Layout>(layout),
                          static_cast<clblast::Triangle>(triangle),
                          static_cast<clblast::Transpose>(a_transpose),
                          static_cast<clblast::Diagonal>(diagonal),
                          n,
                          ap_buffer, ap_offset,
                          x_buffer, x_offset, x_inc,
                          queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TRSV
CLBlastStatusCode CLBlastStrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsv<float>(static_cast<clblast::Layout>(layout),
                           static_cast<clblast::Triangle>(triangle),
                           static_cast<clblast::Transpose>(a_transpose),
                           static_cast<clblast::Diagonal>(diagonal),
                           n,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsv<double>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsv<float2>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsv<double2>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             static_cast<clblast::Transpose>(a_transpose),
                             static_cast<clblast::Diagonal>(diagonal),
                             n,
                             a_buffer, a_offset, a_ld,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TBSV
CLBlastStatusCode CLBlastStbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbsv<float>(static_cast<clblast::Layout>(layout),
                           static_cast<clblast::Triangle>(triangle),
                           static_cast<clblast::Transpose>(a_transpose),
                           static_cast<clblast::Diagonal>(diagonal),
                           n, k,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbsv<double>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n, k,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbsv<float2>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n, k,
                            a_buffer, a_offset, a_ld,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n, const size_t k,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tbsv<double2>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             static_cast<clblast::Transpose>(a_transpose),
                             static_cast<clblast::Diagonal>(diagonal),
                             n, k,
                             a_buffer, a_offset, a_ld,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TPSV
CLBlastStatusCode CLBlastStpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpsv<float>(static_cast<clblast::Layout>(layout),
                           static_cast<clblast::Triangle>(triangle),
                           static_cast<clblast::Transpose>(a_transpose),
                           static_cast<clblast::Diagonal>(diagonal),
                           n,
                           ap_buffer, ap_offset,
                           x_buffer, x_offset, x_inc,
                           queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpsv<double>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            ap_buffer, ap_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpsv<float2>(static_cast<clblast::Layout>(layout),
                            static_cast<clblast::Triangle>(triangle),
                            static_cast<clblast::Transpose>(a_transpose),
                            static_cast<clblast::Diagonal>(diagonal),
                            n,
                            ap_buffer, ap_offset,
                            x_buffer, x_offset, x_inc,
                            queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t n,
                               const cl_mem ap_buffer, const size_t ap_offset,
                               cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Tpsv<double2>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Triangle>(triangle),
                             static_cast<clblast::Transpose>(a_transpose),
                             static_cast<clblast::Diagonal>(diagonal),
                             n,
                             ap_buffer, ap_offset,
                             x_buffer, x_offset, x_inc,
                             queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// GER
CLBlastStatusCode CLBlastSger(const CLBlastLayout layout,
                              const size_t m, const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Ger(static_cast<clblast::Layout>(layout),
                   m, n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDger(const CLBlastLayout layout,
                              const size_t m, const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Ger(static_cast<clblast::Layout>(layout),
                   m, n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHger(const CLBlastLayout layout,
                              const size_t m, const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Ger(static_cast<clblast::Layout>(layout),
                   m, n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// GERU
CLBlastStatusCode CLBlastCgeru(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Geru(static_cast<clblast::Layout>(layout),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZgeru(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Geru(static_cast<clblast::Layout>(layout),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// GERC
CLBlastStatusCode CLBlastCgerc(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gerc(static_cast<clblast::Layout>(layout),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZgerc(const CLBlastLayout layout,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gerc(static_cast<clblast::Layout>(layout),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HER
CLBlastStatusCode CLBlastCher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Her(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Her(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HPR
CLBlastStatusCode CLBlastChpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hpr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   ap_buffer, ap_offset,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZhpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hpr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   ap_buffer, ap_offset,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HER2
CLBlastStatusCode CLBlastCher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Her2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    float2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Her2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    double2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HPR2
CLBlastStatusCode CLBlastChpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_float2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hpr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    float2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    ap_buffer, ap_offset,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZhpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_double2 alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hpr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    double2{alpha.s[0], alpha.s[1]},
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    ap_buffer, ap_offset,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SYR
CLBlastStatusCode CLBlastSsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   a_buffer, a_offset, a_ld,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SPR
CLBlastStatusCode CLBlastSspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const float alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   ap_buffer, ap_offset,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const double alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   ap_buffer, ap_offset,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                              const size_t n,
                              const cl_half alpha,
                              const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                              cl_mem ap_buffer, const size_t ap_offset,
                              cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spr(static_cast<clblast::Layout>(layout),
                   static_cast<clblast::Triangle>(triangle),
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   ap_buffer, ap_offset,
                   queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SYR2
CLBlastStatusCode CLBlastSsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    a_buffer, a_offset, a_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SPR2
CLBlastStatusCode CLBlastSspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const float alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    ap_buffer, ap_offset,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const double alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    ap_buffer, ap_offset,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                               const size_t n,
                               const cl_half alpha,
                               const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                               const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                               cl_mem ap_buffer, const size_t ap_offset,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Spr2(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    n,
                    alpha,
                    x_buffer, x_offset, x_inc,
                    y_buffer, y_offset, y_inc,
                    ap_buffer, ap_offset,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// GEMM
CLBlastStatusCode CLBlastSgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Transpose>(b_transpose),
                    m, n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Transpose>(b_transpose),
                    m, n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Transpose>(b_transpose),
                    m, n, k,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    float2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Transpose>(b_transpose),
                    m, n, k,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    double2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Gemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Transpose>(b_transpose),
                    m, n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SYMM
CLBlastStatusCode CLBlastSsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    float2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    double2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Symm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HEMM
CLBlastStatusCode CLBlastChemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    float2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZhemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Hemm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    double2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SYRK
CLBlastStatusCode CLBlastSsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syrk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syrk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_float2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syrk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    float2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_double2 beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syrk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    double2{beta.s[0], beta.s[1]},
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const cl_half beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syrk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HERK
CLBlastStatusCode CLBlastCherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const float beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Herk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                               const size_t n, const size_t k,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               const double beta,
                               cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Herk(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    beta,
                    c_buffer, c_offset, c_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// SYR2K
CLBlastStatusCode CLBlastSsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const float alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const float beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const double alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const double beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_float2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const cl_float2 beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     float2{alpha.s[0], alpha.s[1]},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     float2{beta.s[0], beta.s[1]},
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_double2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const cl_double2 beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     double2{alpha.s[0], alpha.s[1]},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     double2{beta.s[0], beta.s[1]},
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_half alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const cl_half beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Syr2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     alpha,
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// HER2K
CLBlastStatusCode CLBlastCher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_float2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const float beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Her2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     float2{alpha.s[0], alpha.s[1]},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                const size_t n, const size_t k,
                                const cl_double2 alpha,
                                const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                const double beta,
                                cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Her2k(static_cast<clblast::Layout>(layout),
                     static_cast<clblast::Triangle>(triangle),
                     static_cast<clblast::Transpose>(ab_transpose),
                     n, k,
                     double2{alpha.s[0], alpha.s[1]},
                     a_buffer, a_offset, a_ld,
                     b_buffer, b_offset, b_ld,
                     beta,
                     c_buffer, c_offset, c_ld,
                     queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TRMM
CLBlastStatusCode CLBlastStrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_half alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trmm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// TRSM
CLBlastStatusCode CLBlastStrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const float alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const double alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastCtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_float2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    float2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                               const size_t m, const size_t n,
                               const cl_double2 alpha,
                               const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                               cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                               cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Trsm(static_cast<clblast::Layout>(layout),
                    static_cast<clblast::Side>(side),
                    static_cast<clblast::Triangle>(triangle),
                    static_cast<clblast::Transpose>(a_transpose),
                    static_cast<clblast::Diagonal>(diagonal),
                    m, n,
                    double2{alpha.s[0], alpha.s[1]},
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// OMATCOPY
CLBlastStatusCode CLBlastSomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const float alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Transpose>(a_transpose),
                        m, n,
                        alpha,
                        a_buffer, a_offset, a_ld,
                        b_buffer, b_offset, b_ld,
                        queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastDomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const double alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Transpose>(a_transpose),
                        m, n,
                        alpha,
                        a_buffer, a_offset, a_ld,
                        b_buffer, b_offset, b_ld,
                        queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastComatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_float2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Transpose>(a_transpose),
                        m, n,
                        float2{alpha.s[0], alpha.s[1]},
                        a_buffer, a_offset, a_ld,
                        b_buffer, b_offset, b_ld,
                        queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastZomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_double2 alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Transpose>(a_transpose),
                        m, n,
                        double2{alpha.s[0], alpha.s[1]},
                        a_buffer, a_offset, a_ld,
                        b_buffer, b_offset, b_ld,
                        queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}
CLBlastStatusCode CLBlastHomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                   const size_t m, const size_t n,
                                   const cl_half alpha,
                                   const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                   cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                   cl_command_queue* queue, cl_event* event) {
  try {
    return static_cast<CLBlastStatusCode>(
      clblast::Omatcopy(static_cast<clblast::Layout>(layout),
                        static_cast<clblast::Transpose>(a_transpose),
                        m, n,
                        alpha,
                        a_buffer, a_offset, a_ld,
                        b_buffer, b_offset, b_ld,
                        queue, event)
    );
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================

// Clears the cache of stored binaries
CLBlastStatusCode CLBlastClearCache() {
  try {
    return static_cast<CLBlastStatusCode>(clblast::ClearCache());
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// Fills the cache with binaries for a specific device
CLBlastStatusCode CLBlastFillCache(const cl_device_id device) {
  try {
    return static_cast<CLBlastStatusCode>(clblast::FillCache(device));
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================

// Overrides the tuning parameters for this device-precision-kernel combination
CLBlastStatusCode PUBLIC_API CLBlastOverrideParameters(const cl_device_id device, const char* kernel_name,
                                                       const CLBlastPrecision precision, const size_t num_parameters,
                                                       const char** parameters_names, const size_t* parameters_values) {
  try {
    const auto kernel_name_cpp = std::string(kernel_name);
    const auto precision_cpp = static_cast<clblast::Precision>(precision);
    auto parameters = std::unordered_map<std::string, size_t>();
    for (auto i = size_t{0}; i < num_parameters; ++i) {
      const auto parameter_name = std::string(parameters_names[i]);
      const auto parameter_value = parameters_values[i];
      parameters[parameter_name] = parameter_value;
    }
    const auto status = clblast::OverrideParameters(device, kernel_name_cpp, precision_cpp, parameters);
    return static_cast<CLBlastStatusCode>(status);
  } catch (...) { return static_cast<CLBlastStatusCode>(clblast::DispatchExceptionForC()); }
}

// =================================================================================================
