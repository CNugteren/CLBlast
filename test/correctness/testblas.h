
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file tests any CLBlast routine. It contains two types of tests: one testing all sorts of
// input combinations, and one deliberatly testing with invalid values.
// Typename T: the data-type of the routine's memory buffers (==precision)
// Typename U: the data-type of the alpha and beta arguments
//
// =================================================================================================

#ifndef CLBLAST_TEST_CORRECTNESS_TESTBLAS_H_
#define CLBLAST_TEST_CORRECTNESS_TESTBLAS_H_

#include <vector>
#include <string>

#include "correctness/tester.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class TestBlas: public Tester<T,U> {
 public:

  // Uses several variables from the Tester class
  using Tester<T,U>::context_;
  using Tester<T,U>::queue_;
  using Tester<T,U>::full_test_;
  using Tester<T,U>::device_;

  // Uses several helper functions from the Tester class
  using Tester<T,U>::TestStart;
  using Tester<T,U>::TestEnd;
  using Tester<T,U>::TestErrorCount;
  using Tester<T,U>::TestErrorCodes;
  using Tester<T,U>::GetOffsets;

  // Test settings for the regular test. Append to these lists in case more tests are required.
  const std::vector<size_t> kVectorDims = { 7, 93, 4096 };
  const std::vector<size_t> kIncrements = { 1, 2, 7 };
  const std::vector<size_t> kMatrixDims = { 7, 64 };
  const std::vector<size_t> kMatrixVectorDims = { 61, 512 };
  const std::vector<size_t> kOffsets = GetOffsets();
  const std::vector<U> kAlphaValues = GetExampleScalars<U>(full_test_);
  const std::vector<U> kBetaValues = GetExampleScalars<U>(full_test_);

  // Test settings for the invalid tests
  const std::vector<size_t> kInvalidIncrements = { 0, 1 };
  const size_t kBufferSize = 64;
  const std::vector<size_t> kMatSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
  const std::vector<size_t> kVecSizes = {0, kBufferSize - 1, kBufferSize};

  // The layout/transpose/triangle options to test with
  const std::vector<Layout> kLayouts = {Layout::kRowMajor, Layout::kColMajor};
  const std::vector<Triangle> kTriangles = {Triangle::kUpper, Triangle::kLower};
  const std::vector<Side> kSides = {Side::kLeft, Side::kRight};
  const std::vector<Diagonal> kDiagonals = {Diagonal::kUnit, Diagonal::kNonUnit};
  static const std::vector<Transpose> kTransposes; // Data-type dependent, see .cc-file

  // Shorthand for the routine-specific functions passed to the tester
  using Routine = std::function<StatusCode(const Arguments<U>&, const Buffers&, CommandQueue&)>;
  using ResultGet = std::function<std::vector<T>(const Arguments<U>&, Buffers&, CommandQueue&)>;
  using ResultIndex = std::function<size_t(const Arguments<U>&, const size_t, const size_t)>;
  using ResultIterator = std::function<size_t(const Arguments<U>&)>;

  // Constructor, initializes the base class tester and input data
  TestBlas(int argc, char *argv[], const bool silent,
           const std::string &name, const std::vector<std::string> &options,
           const Routine run_routine, const Routine run_reference, const ResultGet get_result,
           const ResultIndex get_index, const ResultIterator get_id1, const ResultIterator get_id2);

  // The test functions, taking no inputs
  void TestRegular(std::vector<Arguments<U>> &test_vector, const std::string &name);
  void TestInvalid(std::vector<Arguments<U>> &test_vector, const std::string &name);

 private:

  // Source data to test with
  std::vector<T> x_source_;
  std::vector<T> y_source_;
  std::vector<T> a_source_;
  std::vector<T> b_source_;
  std::vector<T> c_source_;
  
  // The routine-specific functions passed to the tester
  Routine run_routine_;
  Routine run_reference_;
  ResultGet get_result_;
  ResultIndex get_index_;
  ResultIterator get_id1_;
  ResultIterator get_id2_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_CORRECTNESS_TESTBLAS_H_
#endif
