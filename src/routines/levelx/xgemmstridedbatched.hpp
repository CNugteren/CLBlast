
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XgemmStridedBatched routine. This is a non-blas batched version of GEMM.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_
#define CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_

#include <vector>

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class XgemmStridedBatched: public Routine {
public:

  // Constructor
  XgemmStridedBatched(Queue &queue, EventPointer event, const std::string &name = "GEMMSTRIDEDBATCHED");

  // Templated-precision implementation of the routine
  void DoGemmStridedBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const size_t m, const size_t n, const size_t k, const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                            const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const T beta,
                            const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                            const size_t batch_count);

  // Indirect version of strided batched GEMM (with pre and post-processing kernels)
  void BatchedGemmIndirect(const size_t m, const size_t n, const size_t k, const T alpha,
                           const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                           const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const T beta,
                           const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                           const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                           const bool a_conjugate, const bool b_conjugate,
                           const size_t a_one, const size_t a_two,
                           const size_t b_one, const size_t b_two,
                           const size_t c_one, const size_t c_two,
                           const size_t batch_count);

  // Direct version of strided batched GEMM (no pre and post-processing kernels)
  void BatchedGemmDirect(const size_t m, const size_t n, const size_t k, const T alpha,
                         const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                         const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride, const T beta,
                         const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                         const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                         const bool a_conjugate, const bool b_conjugate,
                         const size_t batch_count);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_
#endif
