
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the TestXY class (see the header for information about the class).
//
// =================================================================================================

#include <algorithm>

#include "correctness/testxy.h"

namespace clblast {
// =================================================================================================

// Constructor, initializes the base class tester and input data
template <typename T>
TestXY<T>::TestXY(const size_t platform_id, const size_t device_id,
                  const std::string &name, const std::vector<std::string> &options,
                  const Routine clblast_lambda, const Routine clblas_lambda):
    Tester<T>{platform_id, device_id, name, options},
    clblast_lambda_(clblast_lambda),
    clblas_lambda_(clblas_lambda) {

  // Computes the maximum sizes. This allows for a single set of input/output buffers.
  auto max_dim = *std::max_element(kVectorDims.begin(), kVectorDims.end());
  auto max_inc = *std::max_element(kIncrements.begin(), kIncrements.end());
  auto max_offset = *std::max_element(kOffsets.begin(), kOffsets.end());

  // Creates test input data
  x_source_.resize(max_dim*max_inc + max_offset);
  y_source_.resize(max_dim*max_inc + max_offset);
  PopulateVector(x_source_);
  PopulateVector(y_source_);
}

// ===============================================================================================

// Tests the routine for a wide variety of parameters
template <typename T>
void TestXY<T>::TestRegular(Arguments<T> &args, const std::string &name) {
  TestStart("regular behaviour", name);

  // Iterates over the vector dimension
  for (auto &n: kVectorDims) {
    args.n = n;

    // Iterates over the increment-values and the offsets
    for (auto &x_inc: kIncrements) {
      args.x_inc = x_inc;
      for (auto &x_offset: kOffsets) {
        args.x_offset = x_offset;
        for (auto &y_inc: kIncrements) {
          args.y_inc = y_inc;
          for (auto &y_offset: kOffsets) {
            args.y_offset = y_offset;

            // Computes the buffer sizes
            auto x_size = n * x_inc + x_offset;
            auto y_size = n * y_inc + y_offset;
            if (x_size < 1 || y_size < 1) { continue; }

            // Creates the OpenCL buffers
            auto x_vec = Buffer(context_, CL_MEM_READ_WRITE, x_size*sizeof(T));
            auto r_vec = Buffer(context_, CL_MEM_READ_WRITE, y_size*sizeof(T));
            auto s_vec = Buffer(context_, CL_MEM_READ_WRITE, y_size*sizeof(T));

            // Iterates over the values for alpha
            for (auto &alpha: kAlphaValues) {
              args.alpha = alpha;

              // Runs the reference clBLAS code
              x_vec.WriteBuffer(queue_, x_size*sizeof(T), x_source_);
              r_vec.WriteBuffer(queue_, y_size*sizeof(T), y_source_);
              auto status1 = clblas_lambda_(args, x_vec, r_vec, queue_);

              // Runs the CLBlast code
              x_vec.WriteBuffer(queue_, x_size*sizeof(T), x_source_);
              s_vec.WriteBuffer(queue_, y_size*sizeof(T), y_source_);
              auto status2 = clblast_lambda_(args, x_vec, s_vec, queue_);

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
              for (auto idn=size_t{0}; idn<n; ++idn) {
                auto index = idn*y_inc + y_offset;
                if (!TestSimilarity(r_result[index], s_result[index])) {
                  errors++;
                }
              }

              // Tests the error count (should be zero)
              TestErrorCount(errors, n, args);
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
void TestXY<T>::TestInvalidBufferSizes(Arguments<T> &args, const std::string &name) {
  TestStart("invalid buffer sizes", name);

  // Sets example test parameters
  args.n = kBufferSize;
  args.x_offset = 0;
  args.y_offset = 0;

  // Iterates over test buffer sizes
  const std::vector<size_t> kBufferSizes = {0, kBufferSize - 1, kBufferSize};
  for (auto &x_size: kBufferSizes) {
    for (auto &y_size: kBufferSizes) {

      // Iterates over test increments
      for (auto &x_inc: kInvalidIncrements) {
        args.x_inc = x_inc;
        for (auto &y_inc: kInvalidIncrements) {
          args.y_inc = y_inc;

          // Creates the OpenCL buffers. Note: we are not using the C++ version since we explicitly
          // want to be able to create invalid buffers (no error checking here).
          auto x = clCreateBuffer(context_(), CL_MEM_READ_WRITE, x_size*sizeof(T), nullptr, nullptr);
          auto x_vec = Buffer(x);
          auto r = clCreateBuffer(context_(), CL_MEM_READ_WRITE, y_size*sizeof(T), nullptr, nullptr);
          auto r_vec = Buffer(r);
          auto s = clCreateBuffer(context_(), CL_MEM_READ_WRITE, y_size*sizeof(T), nullptr, nullptr);
          auto s_vec = Buffer(s);

          // Runs the two routines
          auto status1 = clblas_lambda_(args, x_vec, r_vec, queue_);
          auto status2 = clblast_lambda_(args, x_vec, s_vec, queue_);

          // Tests for equality of the two status codes
          TestErrorCodes(status1, status2, args);
        }
      }
    }
  }
  TestEnd();
}

// =================================================================================================

// Compiles the templated class
template class TestXY<float>;
template class TestXY<double>;
template class TestXY<float2>;
template class TestXY<double2>;

// =================================================================================================
} // namespace clblast
