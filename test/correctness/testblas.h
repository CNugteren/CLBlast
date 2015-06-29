
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
template <typename T>
class TestBlas: public Tester<T> {
 public:

  // Uses several variables from the Tester class
  using Tester<T>::context_;
  using Tester<T>::queue_;

  // Uses several helper functions from the Tester class
  using Tester<T>::TestStart;
  using Tester<T>::TestEnd;
  using Tester<T>::TestSimilarity;
  using Tester<T>::TestErrorCount;
  using Tester<T>::TestErrorCodes;
  using Tester<T>::GetExampleScalars;
  using Tester<T>::GetOffsets;
  using Tester<T>::PrecisionSupported;

  // Test settings for the regular test. Append to these lists in case more tests are required.
  const std::vector<size_t> kVectorDims = { 7, 93, 4096 };
  const std::vector<size_t> kIncrements = { 1, 2, 7 };
  const std::vector<size_t> kMatrixDims = { 7, 64 };
  const std::vector<size_t> kMatrixVectorDims = { 61, 512 };
  const std::vector<size_t> kOffsets = GetOffsets();
  const std::vector<T> kAlphaValues = GetExampleScalars();
  const std::vector<T> kBetaValues = GetExampleScalars();

  // Test settings for the invalid tests
  const std::vector<size_t> kInvalidIncrements = { 0, 1 };
  const size_t kBufferSize = 64;
  const std::vector<size_t> kMatSizes = {0, kBufferSize*kBufferSize-1, kBufferSize*kBufferSize};
  const std::vector<size_t> kVecSizes = {0, kBufferSize - 1, kBufferSize};

  // The layout/transpose/triangle options to test with
  const std::vector<Layout> kLayouts = {Layout::kRowMajor, Layout::kColMajor};
  const std::vector<Triangle> kTriangles = {Triangle::kUpper, Triangle::kLower};
  const std::vector<Side> kSides = {Side::kLeft, Side::kRight};
  static const std::vector<Transpose> kTransposes; // Data-type dependent, see .cc-file

  // Shorthand for the routine-specific functions passed to the tester
  using Routine = std::function<StatusCode(const Arguments<T>&, const Buffers&, CommandQueue&)>;
  using ResultGet = std::function<std::vector<T>(const Arguments<T>&, Buffers&, CommandQueue&)>;
  using ResultIndex = std::function<size_t(const Arguments<T>&, const size_t, const size_t)>;
  using ResultIterator = std::function<size_t(const Arguments<T>&)>;

  // Constructor, initializes the base class tester and input data
  TestBlas(int argc, char *argv[], const bool silent,
           const std::string &name, const std::vector<std::string> &options,
           const Routine run_routine, const Routine run_reference, const ResultGet get_result,
           const ResultIndex get_index, const ResultIterator get_id1, const ResultIterator get_id2);

  // The test functions, taking no inputs
  void TestRegular(std::vector<Arguments<T>> &test_vector, const std::string &name);
  void TestInvalid(std::vector<Arguments<T>> &test_vector, const std::string &name);

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
