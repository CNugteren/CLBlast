
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XgemmBatched routine. This is a non-blas batched version of GEMM.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMMBATCHED_H_
#define CLBLAST_ROUTINES_XGEMMBATCHED_H_

#include <cstddef>
#include <string>
#include <vector>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class XgemmBatched : public Routine {
 public:
  // Constructor
  XgemmBatched(Queue& queue, EventPointer event, const std::string& name = "GEMMBATCHED");

  // Templated-precision implementation of the routine
  void DoGemmBatched(Layout layout, Transpose a_transpose, Transpose b_transpose, size_t m, size_t n, size_t k,
                     const std::vector<T>& alphas, const Buffer<T>& a_buffer, const std::vector<size_t>& a_offsets,
                     size_t a_ld, const Buffer<T>& b_buffer, const std::vector<size_t>& b_offsets, size_t b_ld,
                     const std::vector<T>& betas, const Buffer<T>& c_buffer, const std::vector<size_t>& c_offsets,
                     size_t c_ld, size_t batch_count);

  // Indirect version of batched GEMM (with pre and post-processing kernels)
  void BatchedGemmIndirect(size_t m, size_t n, size_t k, const Buffer<T>& alphas, const Buffer<T>& a_buffer,
                           const std::vector<int>& a_offsets, size_t a_ld, const Buffer<T>& b_buffer,
                           const std::vector<int>& b_offsets, size_t b_ld, const Buffer<T>& betas,
                           const Buffer<T>& c_buffer, const std::vector<int>& c_offsets, size_t c_ld,
                           bool a_do_transpose, bool b_do_transpose, bool c_do_transpose, bool a_conjugate,
                           bool b_conjugate, size_t a_one, size_t a_two, size_t b_one, size_t b_two, size_t c_one,
                           size_t c_two, size_t batch_count);

  // Direct version of batched GEMM (no pre and post-processing kernels)
  void BatchedGemmDirect(size_t m, size_t n, size_t k, const Buffer<T>& alphas, const Buffer<T>& a_buffer,
                         const std::vector<int>& a_offsets, size_t a_ld, const Buffer<T>& b_buffer,
                         const std::vector<int>& b_offsets, size_t b_ld, const Buffer<T>& betas,
                         const Buffer<T>& c_buffer, const std::vector<int>& c_offsets, size_t c_ld, bool a_do_transpose,
                         bool b_do_transpose, bool c_do_transpose, bool a_conjugate, bool b_conjugate,
                         size_t batch_count);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGEMMBATCHED_H_
#endif
