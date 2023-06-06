
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
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
class Xgemm: public Routine {
 public:

  // Defines the assumptions of the GEMM kernels
  static bool a_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }
  static bool b_want_rotated_(const size_t) { return true; }
  static bool c_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }

  // Computes the size of the temporary GEMM buffer based on user-arguments
  static size_t GetTempSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                            const size_t m, const size_t n, const size_t k,
                            const size_t a_offset, const size_t a_ld,
                            const size_t b_offset, const size_t b_ld,
                            const size_t c_offset, const size_t c_ld,
                            const size_t mwg, const size_t nwg, const size_t kwg,
                            const size_t gemm_kernel_id) {

    // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
    bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    ProcessArguments(layout, a_transpose, b_transpose, m, n, k,
                     a_one, a_two, b_one, b_two, c_one, c_two,
                     a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                     gemm_kernel_id);

    // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
    // whether the matrices need to be rotated or not for the kernel.
    size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
    CalculateInternalDimensions(m, n, k, mwg, nwg, kwg,
                                a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                                gemm_kernel_id);

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
    auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
    auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

    // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
    auto b_temp_offset = size_t{0};
    auto c_temp_offset = size_t{0};
    return ComputeTempSize(a_no_temp, b_no_temp, c_no_temp,
                           a_one_i*a_two_i, b_one_i*b_two_i, c_one_i*c_two_i,
                           b_temp_offset, c_temp_offset);
  }

  // Selects which version of GEMM to run
  static bool UseDirectKernel(const size_t m, const size_t n, const size_t k,
                              const size_t min_indirect_size) {
    const auto m_n_k = static_cast<unsigned long long>(m) * static_cast<unsigned long long>(n) *
                       static_cast<unsigned long long>(k);
    const auto min_indirect_size_ll = static_cast<unsigned long long>(min_indirect_size);
    const auto min_indirect_size_e3 = min_indirect_size_ll * min_indirect_size_ll * min_indirect_size_ll;
    return (m_n_k < min_indirect_size_e3);
  }

  // Process the user-arguments, computes secondary parameters
  static void ProcessArguments(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               size_t& a_one, size_t& a_two, size_t& b_one,
                               size_t& b_two, size_t& c_one, size_t& c_two,
                               bool& a_do_transpose, bool& b_do_transpose, bool& c_do_transpose,
                               bool& a_conjugate, bool& b_conjugate,
                               const size_t gemm_kernel_id) {

    // Makes sure all dimensions are larger than zero
    if ((m == 0) || (n == 0) || (k == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

    // Computes whether or not the matrices are transposed in memory. This is based on their layout
    // (row or column-major) and whether or not they are requested to be pre-transposed. Note
    // that the Xgemm kernel expects either matrices A and C (in case of row-major) or B (in case of
    // col-major) to be transformed, so transposing requirements are not the same as whether or not
    // the matrix is actually transposed in memory.
    const auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                           (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
    const auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                           (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
    const auto c_rotated = (layout == Layout::kRowMajor);
    a_do_transpose = a_rotated != a_want_rotated_(gemm_kernel_id);
    b_do_transpose = b_rotated != b_want_rotated_(gemm_kernel_id);
    c_do_transpose = c_rotated != c_want_rotated_(gemm_kernel_id);

    // In case of complex data-types, the transpose can also become a conjugate transpose
    a_conjugate = (a_transpose == Transpose::kConjugate);
    b_conjugate = (b_transpose == Transpose::kConjugate);

    // Computes the first and second dimensions of the 3 matrices taking into account whether the
    // matrices are rotated or not
    a_one = (a_rotated) ? k : m;
    a_two = (a_rotated) ? m : k;
    b_one = (b_rotated) ? n : k;
    b_two = (b_rotated) ? k : n;
    c_one = (c_rotated) ? n : m;
    c_two = (c_rotated) ? m : n;
  }

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  static size_t ComputeTempSize(const bool a_no_temp, const bool b_no_temp, const bool c_no_temp,
                                const size_t a_size, const size_t b_size, const size_t c_size,
                                size_t &b_temp_offset, size_t &c_temp_offset) {
    auto temp_size = size_t{0};
    if (!a_no_temp) { temp_size += a_size; }
    if (!b_no_temp) { b_temp_offset = temp_size; temp_size += b_size; }
    if (!c_no_temp) { c_temp_offset = temp_size; temp_size += c_size; }
    return temp_size;
  }

  // Determines whether or not temporary matrices are needed
  static bool NoTempBuffer(const size_t one, const size_t one_i, const size_t two, const size_t two_i,
                           const size_t ld, const size_t offset,
                           const bool do_transpose, const bool conjugate) {
    return one == one_i && two == two_i && ld == one && offset == 0 && !do_transpose && !conjugate;
  }


  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  static void CalculateInternalDimensions(const size_t m, const size_t n, const size_t k,
                                          const size_t mwg, const size_t nwg, const size_t kwg,
                                          size_t& a_one_i, size_t& a_two_i, size_t& b_one_i,
                                          size_t& b_two_i, size_t& c_one_i, size_t& c_two_i,
                                          const size_t gemm_kernel_id) {
    const auto m_ceiled = Ceil(m, mwg);
    const auto n_ceiled = Ceil(n, nwg);
    const auto k_ceiled = Ceil(k, kwg);
    a_one_i = (a_want_rotated_(gemm_kernel_id)) ? k_ceiled : m_ceiled;
    a_two_i = (a_want_rotated_(gemm_kernel_id)) ? m_ceiled : k_ceiled;
    b_one_i = (b_want_rotated_(gemm_kernel_id)) ? n_ceiled : k_ceiled;
    b_two_i = (b_want_rotated_(gemm_kernel_id)) ? k_ceiled : n_ceiled;
    c_one_i = (c_want_rotated_(gemm_kernel_id)) ? n_ceiled : m_ceiled;
    c_two_i = (c_want_rotated_(gemm_kernel_id)) ? m_ceiled : n_ceiled;
  }

  // Constructor
  Xgemm(Queue &queue, EventPointer event, const std::string &name = "GEMM");

  // Templated-precision implementation of the routine
  void DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
              const size_t m, const size_t n, const size_t k,
              const T alpha,
              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
              const T beta,
              const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
              const Buffer<T> &temp_buffer = Buffer<T>(0), const bool temp_buffer_provided = false);

  // Indirect version of GEMM (with pre and post-processing kernels)
  void GemmIndirect(const size_t m, const size_t n, const size_t k,
                    const T alpha,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                    const T beta,
                    const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                    const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                    const bool a_conjugate, const bool b_conjugate,
                    const size_t a_one, const size_t a_two,
                    const size_t b_one, const size_t b_two,
                    const size_t c_one, const size_t c_two,
                    const Buffer<T> &temp_buffer, const bool temp_buffer_provided);

  // Direct version of GEMM (no pre and post-processing kernels)
  void GemmDirect(const size_t m, const size_t n, const size_t k,
                  const T alpha,
                  const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                  const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                  const T beta,
                  const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                  const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                  const bool a_conjugate, const bool b_conjugate);
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XGEMM_H_
#endif
