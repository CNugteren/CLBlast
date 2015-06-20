
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the TestABC class (see the header for information about the class).
//
// =================================================================================================

#include <algorithm>

#include "correctness/testabc.h"

namespace clblast {
// =================================================================================================

// Constructor, initializes the base class tester and input data
template <typename T>
TestABC<T>::TestABC(int argc, char *argv[], const bool silent,
                    const std::string &name, const std::vector<std::string> &options,
                    const Routine clblast_lambda, const Routine clblas_lambda):
    Tester<T>{argc, argv, silent, name, options},
    clblast_lambda_(clblast_lambda),
    clblas_lambda_(clblas_lambda) {

  // Computes the maximum sizes. This allows for a single set of input/output buffers.
  auto max_dim = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  auto max_ld = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  auto max_offset = *std::max_element(kOffsets.begin(), kOffsets.end());

  // Creates test input data
  a_source_.resize(max_dim*max_ld + max_offset);
  b_source_.resize(max_dim*max_ld + max_offset);
  c_source_.resize(max_dim*max_ld + max_offset);
  PopulateVector(a_source_);
  PopulateVector(b_source_);
  PopulateVector(c_source_);
}

// ===============================================================================================

// Tests the routine for a wide variety of parameters
template <typename T>
void TestABC<T>::TestRegular(Arguments<T> &args, const std::string &name) {
  if (!PrecisionSupported()) { return; }
  TestStart("regular behaviour", name);

  // Computes whether or not the matrices are transposed. Note that we assume a default of
  // column-major and no-transpose. If one of them is different (but not both), then rotated
  // is considered true.
  auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                   (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
  auto b_rotated = (args.layout == Layout::kColMajor && args.b_transpose != Transpose::kNo) ||
                   (args.layout == Layout::kRowMajor && args.b_transpose == Transpose::kNo);
  auto c_rotated = (args.layout == Layout::kRowMajor);

  // Iterates over the matrix dimensions
  for (auto &m: kMatrixDims) {
    args.m = m;
    for (auto &n: kMatrixDims) {
      args.n = n;
      for (auto &k: kMatrixDims) {
        args.k = k;

        // Computes the second dimensions of the matrices taking the rotation into account
        auto a_two = (a_rotated) ? m : k;
        auto b_two = (b_rotated) ? k : n;
        auto c_two = (c_rotated) ? m : n;

        // Iterates over the leading-dimension values and the offsets
        for (auto &a_ld: kMatrixDims) {
          args.a_ld = a_ld;
          for (auto &a_offset: kOffsets) {
            args.a_offset = a_offset;
            for (auto &b_ld: kMatrixDims) {
              args.b_ld = b_ld;
              for (auto &b_offset: kOffsets) {
                args.b_offset = b_offset;
                for (auto &c_ld: kMatrixDims) {
                  args.c_ld = c_ld;
                  for (auto &c_offset: kOffsets) {
                    args.c_offset = c_offset;

                    // Computes the buffer sizes
                    auto a_size = a_two * a_ld + a_offset;
                    auto b_size = b_two * b_ld + b_offset;
                    auto c_size = c_two * c_ld + c_offset;
                    if (a_size < 1 || b_size < 1 || c_size < 1) { continue; }

                    // Creates the OpenCL buffers
                    auto a_mat = Buffer(context_, CL_MEM_READ_WRITE, a_size*sizeof(T));
                    auto b_mat = Buffer(context_, CL_MEM_READ_WRITE, b_size*sizeof(T));
                    auto r_mat = Buffer(context_, CL_MEM_READ_WRITE, c_size*sizeof(T));
                    auto s_mat = Buffer(context_, CL_MEM_READ_WRITE, c_size*sizeof(T));

                    // Iterates over the values for alpha and beta
                    for (auto &alpha: kAlphaValues) {
                      args.alpha = alpha;
                      for (auto &beta: kBetaValues) {
                        args.beta = beta;

                        // Runs the reference clBLAS code
                        a_mat.WriteBuffer(queue_, a_size*sizeof(T), a_source_);
                        b_mat.WriteBuffer(queue_, b_size*sizeof(T), b_source_);
                        r_mat.WriteBuffer(queue_, c_size*sizeof(T), c_source_);
                        auto status1 = clblas_lambda_(args, a_mat, b_mat, r_mat, queue_);

                        // Runs the CLBlast code
                        a_mat.WriteBuffer(queue_, a_size*sizeof(T), a_source_);
                        b_mat.WriteBuffer(queue_, b_size*sizeof(T), b_source_);
                        s_mat.WriteBuffer(queue_, c_size*sizeof(T), c_source_);
                        auto status2 = clblast_lambda_(args, a_mat, b_mat, s_mat, queue_);

                        // Tests for equality of the two status codes
                        if (status1 != StatusCode::kSuccess || status2 != StatusCode::kSuccess) {
                          TestErrorCodes(status1, status2, args);
                          continue;
                        }

                        // Downloads the results
                        std::vector<T> r_result(c_size, static_cast<T>(0));
                        std::vector<T> s_result(c_size, static_cast<T>(0));
                        r_mat.ReadBuffer(queue_, c_size*sizeof(T), r_result);
                        s_mat.ReadBuffer(queue_, c_size*sizeof(T), s_result);

                        // Checks for differences in the output
                        auto errors = size_t{0};
                        for (auto idm=size_t{0}; idm<m; ++idm) {
                          for (auto idn=size_t{0}; idn<n; ++idn) {
                            auto index = (args.layout == Layout::kRowMajor) ?
                                          idm*args.c_ld + idn + args.c_offset:
                                          idn*args.c_ld + idm + args.c_offset;
                            if (!TestSimilarity(r_result[index], s_result[index])) {
                              errors++;
                            }
                          }
                        }

                        // Tests the error count (should be zero)
                        TestErrorCount(errors, m*n, args);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  TestEnd();
}

// =================================================================================================

// Tests the routine for cases with invalid OpenCL memory buffer sizes. Tests only on return-types,
// does not test for results (if any).
template <typename T>
void TestABC<T>::TestInvalidBufferSizes(Arguments<T> &args, const std::string &name) {
  if (!PrecisionSupported()) { return; }
  TestStart("invalid buffer sizes", name);

  // Sets example test parameters
  args.m = kBufferSize;
  args.n = kBufferSize;
  args.k = kBufferSize;
  args.a_ld = kBufferSize;
  args.b_ld = kBufferSize;
  args.c_ld = kBufferSize;
  args.a_offset = 0;
  args.b_offset = 0;
  args.c_offset = 0;

  // Iterates over test buffer sizes
  const std::vector<size_t> kBufferSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
  for (auto &a_size: kBufferSizes) {
    for (auto &b_size: kBufferSizes) {
      for (auto &c_size: kBufferSizes) {

        // Creates the OpenCL buffers. Note: we are not using the C++ version since we explicitly
        // want to be able to create invalid buffers (no error checking here).
        auto a = clCreateBuffer(context_(), CL_MEM_READ_WRITE, a_size*sizeof(T), nullptr, nullptr);
        auto a_mat = Buffer(a);
        auto b = clCreateBuffer(context_(), CL_MEM_READ_WRITE, b_size*sizeof(T), nullptr, nullptr);
        auto b_mat = Buffer(b);
        auto r = clCreateBuffer(context_(), CL_MEM_READ_WRITE, c_size*sizeof(T), nullptr, nullptr);
        auto r_mat = Buffer(r);
        auto s = clCreateBuffer(context_(), CL_MEM_READ_WRITE, c_size*sizeof(T), nullptr, nullptr);
        auto s_mat = Buffer(s);

        // Runs the two routines
        auto status1 = clblas_lambda_(args, a_mat, b_mat, r_mat, queue_);
        auto status2 = clblast_lambda_(args, a_mat, b_mat, s_mat, queue_);

        // Tests for equality of the two status codes
        TestErrorCodes(status1, status2, args);
      }
    }
  }
  TestEnd();
}

// =================================================================================================

// Compiles the templated class
template class TestABC<float>;
template class TestABC<double>;
template class TestABC<float2>;
template class TestABC<double2>;

// =================================================================================================
} // namespace clblast
