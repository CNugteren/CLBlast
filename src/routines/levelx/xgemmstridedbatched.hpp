
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XgemmStridedBatched routine. This is a non-blas batched version of GEMM.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_
#define CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class XgemmStridedBatched : public Routine {
 public:
  // Constructor
  XgemmStridedBatched(Queue& queue, EventPointer event, const std::string& name = "GEMMSTRIDEDBATCHED");

  // Templated-precision implementation of the routine
  void DoGemmStridedBatched(Layout layout, Transpose a_transpose, Transpose b_transpose, size_t m, size_t n, size_t k,
                            T alpha, const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld, size_t a_stride,
                            const Buffer<T>& b_buffer, size_t b_offset, size_t b_ld, size_t b_stride, T beta,
                            const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld, size_t c_stride,
                            size_t batch_count);

  // Indirect version of strided batched GEMM (with pre and post-processing kernels)
  void BatchedGemmIndirect(size_t m, size_t n, size_t k, T alpha, const Buffer<T>& a_buffer, size_t a_offset,
                           size_t a_ld, size_t a_stride, const Buffer<T>& b_buffer, size_t b_offset, size_t b_ld,
                           size_t b_stride, T beta, const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld,
                           size_t c_stride, bool a_do_transpose, bool b_do_transpose, bool c_do_transpose,
                           bool a_conjugate, bool b_conjugate, size_t a_one, size_t a_two, size_t b_one, size_t b_two,
                           size_t c_one, size_t c_two, size_t batch_count);

  // Direct version of strided batched GEMM (no pre and post-processing kernels)
  void BatchedGemmDirect(size_t m, size_t n, size_t k, T alpha, const Buffer<T>& a_buffer, size_t a_offset, size_t a_ld,
                         size_t a_stride, const Buffer<T>& b_buffer, size_t b_offset, size_t b_ld, size_t b_stride,
                         T beta, const Buffer<T>& c_buffer, size_t c_offset, size_t c_ld, size_t c_stride,
                         bool a_do_transpose, bool b_do_transpose, bool c_do_transpose, bool a_conjugate,
                         bool b_conjugate, size_t batch_count);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_
#endif
