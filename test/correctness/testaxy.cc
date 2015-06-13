
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the TestAXY class (see the header for information about the class).
//
// =================================================================================================

#include <algorithm>

#include "correctness/testaxy.h"

namespace clblast {
// =================================================================================================

// Constructor, initializes the base class tester and input data
template <typename T>
TestAXY<T>::TestAXY(const size_t platform_id, const size_t device_id,
                    const std::string &name, const std::vector<std::string> &options,
                    const Routine clblast_lambda, const Routine clblas_lambda):
    Tester<T>{platform_id, device_id, name, options},
    clblast_lambda_(clblast_lambda),
    clblas_lambda_(clblas_lambda) {

  // Computes the maximum sizes. This allows for a single set of input/output buffers.
  auto max_dim = *std::max_element(kMatrixVectorDims.begin(), kMatrixVectorDims.end());
  auto max_ld = *std::max_element(kMatrixVectorDims.begin(), kMatrixVectorDims.end());
  auto max_inc = *std::max_element(kIncrements.begin(), kIncrements.end());
  auto max_offset = *std::max_element(kOffsets.begin(), kOffsets.end());

  // Creates test input data
  a_source_.resize(max_dim*max_ld + max_offset);
  x_source_.resize(max_dim*max_inc + max_offset);
  y_source_.resize(max_dim*max_inc + max_offset);
  PopulateVector(a_source_);
  PopulateVector(x_source_);
  PopulateVector(y_source_);
}

// ===============================================================================================

// Tests the routine for a wide variety of parameters
template <typename T>
void TestAXY<T>::TestRegular(Arguments<T> &args, const std::string &name) {
  TestStart("regular behaviour", name);

  // Iterates over the dimension for the matrix and vectors
  for (auto &m: kMatrixVectorDims) {
    args.m = m;
    for (auto &n: kMatrixVectorDims) {
      args.n = n;

      // Computes the second dimension of the matrix taking the rotation into account
      auto a_two = (args.layout == Layout::kRowMajor) ? args.m : args.n;

      // Computes the vector sizes in case the matrix is transposed
      auto a_transposed = (args.a_transpose == Transpose::kYes);
      auto m_real = (a_transposed) ? n : m;
      auto n_real = (a_transposed) ? m : n;

      // Iterates over the leading-dimension values and the offsets of the matrix
      for (auto &a_ld: kMatrixVectorDims) {
        args.a_ld = a_ld;
        for (auto &a_offset: kOffsets) {
          args.a_offset = a_offset;

          // Iterates over the increment-values and the offsets of the vectors
          for (auto &x_inc: kIncrements) {
            args.x_inc = x_inc;
            for (auto &x_offset: kOffsets) {
              args.x_offset = x_offset;
              for (auto &y_inc: kIncrements) {
                args.y_inc = y_inc;
                for (auto &y_offset: kOffsets) {
                  args.y_offset = y_offset;

                  // Computes the buffer sizes
                  auto a_size = a_two * a_ld + a_offset;
                  auto x_size = n_real * x_inc + x_offset;
                  auto y_size = m_real * y_inc + y_offset;
                  if (a_size < 1 || x_size < 1 || y_size < 1) { continue; }

                  // Creates the OpenCL buffers
                  auto a_mat = Buffer(context_, CL_MEM_READ_WRITE, a_size*sizeof(T));
                  auto x_vec = Buffer(context_, CL_MEM_READ_WRITE, x_size*sizeof(T));
                  auto r_vec = Buffer(context_, CL_MEM_READ_WRITE, y_size*sizeof(T));
                  auto s_vec = Buffer(context_, CL_MEM_READ_WRITE, y_size*sizeof(T));

                  // Iterates over the values for alpha and beta
                  for (auto &alpha: kAlphaValues) {
                    args.alpha = alpha;
                    for (auto &beta: kBetaValues) {
                      args.beta = beta;

                      // Runs the reference clBLAS code
                      a_mat.WriteBuffer(queue_, a_size*sizeof(T), a_source_);
                      x_vec.WriteBuffer(queue_, x_size*sizeof(T), x_source_);
                      r_vec.WriteBuffer(queue_, y_size*sizeof(T), y_source_);
                      auto status1 = clblas_lambda_(args, a_mat, x_vec, r_vec, queue_);

                      // Runs the CLBlast code
                      a_mat.WriteBuffer(queue_, a_size*sizeof(T), a_source_);
                      x_vec.WriteBuffer(queue_, x_size*sizeof(T), x_source_);
                      s_vec.WriteBuffer(queue_, y_size*sizeof(T), y_source_);
                      auto status2 = clblast_lambda_(args, a_mat, x_vec, s_vec, queue_);

                      // Tests for equality of the two status codes
                      if (status1 != StatusCode::kSuccess || status2 != StatusCode::kSuccess) {
                        TestErrorCodes(status1, status2, args);
                        continue;
                      }

                      // Downloads the results
                      std::vector<T> r_result(y_size, static_cast<T>(0));
                      std::vector<T> s_result(y_size, static_cast<T>(0));
                      r_vec.ReadBuffer(queue_, y_size*sizeof(T), r_result);
                      s_vec.ReadBuffer(queue_, y_size*sizeof(T), s_result);

                      // Checks for differences in the output
                      auto errors = size_t{0};
                      for (auto idm=size_t{0}; idm<m_real; ++idm) {
                        auto index = idm*y_inc + y_offset;
                        if (!TestSimilarity(r_result[index], s_result[index], kErrorMargin)) {
                          errors++;
                        }
                      }

                      // Tests the error count (should be zero)
                      TestErrorCount(errors, m_real, args);
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
void TestAXY<T>::TestInvalidBufferSizes(Arguments<T> &args, const std::string &name) {
  TestStart("invalid buffer sizes", name);

  // Sets example test parameters
  args.m = kBufferSize;
  args.n = kBufferSize;
  args.a_ld = kBufferSize;
  args.a_offset = 0;
  args.x_offset = 0;
  args.y_offset = 0;

  // Iterates over test buffer sizes
  const std::vector<size_t> kMatrixSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
  const std::vector<size_t> kVectorSizes = {0, kBufferSize - 1, kBufferSize};
  for (auto &a_size: kMatrixSizes) {
    for (auto &x_size: kVectorSizes) {
      for (auto &y_size: kVectorSizes) {

        // Iterates over test increments
        for (auto &x_inc: kInvalidIncrements) {
          args.x_inc = x_inc;
          for (auto &y_inc: kInvalidIncrements) {
            args.y_inc = y_inc;

            // Creates the OpenCL buffers. Note: we are not using the C++ version since we
            // explicitly want to be able to create invalid buffers (no error checking here).
            auto a = clCreateBuffer(context_(), CL_MEM_READ_WRITE, a_size*sizeof(T), nullptr, nullptr);
            auto a_mat = Buffer(a);
            auto x = clCreateBuffer(context_(), CL_MEM_READ_WRITE, x_size*sizeof(T), nullptr, nullptr);
            auto x_vec = Buffer(x);
            auto r = clCreateBuffer(context_(), CL_MEM_READ_WRITE, y_size*sizeof(T), nullptr, nullptr);
            auto r_vec = Buffer(r);
            auto s = clCreateBuffer(context_(), CL_MEM_READ_WRITE, y_size*sizeof(T), nullptr, nullptr);
            auto s_vec = Buffer(s);

            // Runs the two routines
            auto status1 = clblas_lambda_(args, a_mat, x_vec, r_vec, queue_);
            auto status2 = clblast_lambda_(args, a_mat, x_vec, s_vec, queue_);

            // Tests for equality of the two status codes
            TestErrorCodes(status1, status2, args);
          }
        }
      }
    }
  }
  TestEnd();
}

// =================================================================================================

// Compiles the templated class
template class TestAXY<float>;
template class TestAXY<double>;
template class TestAXY<float2>;
template class TestAXY<double2>;

// =================================================================================================
} // namespace clblast
