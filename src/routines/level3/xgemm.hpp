
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMM_H_
#define CLBLAST_ROUTINES_XGEMM_H_

#include "routine.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgemm : public Routine {
 public:
  // Defines the assumptions of the GEMM kernels
  static bool a_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }
  static bool b_want_rotated_(const size_t) { return true; }
  static bool c_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }

  // Computes the size of the temporary GEMM buffer based on user-arguments
  static size_t GetTempSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const size_t m, const size_t n, const size_t k, const size_t a_offset, const size_t a_ld,
                            const size_t b_offset, const size_t b_ld, const size_t c_offset, const size_t c_ld,
                            const size_t mwg, const size_t nwg, const size_t kwg, const size_t gemm_kernel_id);

  // Selects which version of GEMM to run
  static bool UseDirectKernel(const size_t m, const size_t n, const size_t k, const size_t min_indirect_size);

  // Process the user-arguments, computes secondary parameters
  static void ProcessArguments(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                               const size_t m, const size_t n, const size_t k, size_t& a_one, size_t& a_two,
                               size_t& b_one, size_t& b_two, size_t& c_one, size_t& c_two, bool& a_do_transpose,
                               bool& b_do_transpose, bool& c_do_transpose, bool& a_conjugate, bool& b_conjugate,
                               const size_t gemm_kernel_id);

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  static size_t ComputeTempSize(const bool a_no_temp, const bool b_no_temp, const bool c_no_temp, const size_t a_size,
                                const size_t b_size, const size_t c_size, size_t& b_temp_offset, size_t& c_temp_offset);

  // Determines whether or not temporary matrices are needed
  static bool NoTempBuffer(const size_t one, const size_t one_i, const size_t two, const size_t two_i, const size_t ld,
                           const size_t offset, const bool do_transpose, const bool conjugate) {
    return one == one_i && two == two_i && ld == one && offset == 0 && !do_transpose && !conjugate;
  }

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  static void CalculateInternalDimensions(const size_t m, const size_t n, const size_t k, const size_t mwg,
                                          const size_t nwg, const size_t kwg, size_t& a_one_i, size_t& a_two_i,
                                          size_t& b_one_i, size_t& b_two_i, size_t& c_one_i, size_t& c_two_i,
                                          const size_t gemm_kernel_id);

  // Constructor
  Xgemm(Queue& queue, EventPointer event, const std::string& name = "GEMM");

  // Templated-precision implementation of the routine
  void DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const size_t m,
              const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer, const size_t a_offset,
              const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
              const Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
              const Buffer<T>& temp_buffer = Buffer<T>(0), const bool temp_buffer_provided = false);

  // Indirect version of GEMM (with pre and post-processing kernels)
  void GemmIndirect(const size_t m, const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer,
                    const size_t a_offset, const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset,
                    const size_t b_ld, const T beta, const Buffer<T>& c_buffer, const size_t c_offset,
                    const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                    const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
                    const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two,
                    const Buffer<T>& temp_buffer, const bool temp_buffer_provided);

  // Direct version of GEMM (no pre and post-processing kernels)
  void GemmDirect(const size_t m, const size_t n, const size_t k, const T alpha, const Buffer<T>& a_buffer,
                  const size_t a_offset, const size_t a_ld, const Buffer<T>& b_buffer, const size_t b_offset,
                  const size_t b_ld, const T beta, const Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
                  const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                  const bool a_conjugate, const bool b_conjugate);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGEMM_H_
#endif
