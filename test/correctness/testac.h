
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file tests any mat-mat (A,C) routine. It contains two types of tests: one testing
// all sorts of input combinations, and one deliberatly testing with invalid values.
//
// =================================================================================================

#ifndef CLBLAST_TEST_CORRECTNESS_TESTAC_H_
#define CLBLAST_TEST_CORRECTNESS_TESTAC_H_

#include <vector>
#include <string>

#include "correctness/tester.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class TestAC: public Tester<T> {
 public:

  // Uses several variables from the Tester class
  using Tester<T>::context_;
  using Tester<T>::queue_;
  using Tester<T>::kLayouts;
  using Tester<T>::kTransposes;

  // Uses several helper functions from the Tester class
  using Tester<T>::TestStart;
  using Tester<T>::TestEnd;
  using Tester<T>::TestSimilarity;
  using Tester<T>::TestErrorCount;
  using Tester<T>::TestErrorCodes;
  using Tester<T>::GetExampleScalars;
  using Tester<T>::GetOffsets;
  using Tester<T>::PrecisionSupported;

  // Test settings for the regular test. Append to this list in case more tests are required.
  const std::vector<size_t> kMatrixDims = { 7, 64 };
  const std::vector<size_t> kOffsets = GetOffsets();
  const std::vector<T> kAlphaValues = GetExampleScalars();
  const std::vector<T> kBetaValues = GetExampleScalars();

  // Test settings for the invalid test
  const size_t kBufferSize = 64;

  // Shorthand for a BLAS routine
  using Routine = std::function<StatusCode(const Arguments<T>&,
                                           const Buffer&, const Buffer&,
                                           CommandQueue&)>;

  // Constructor, initializes the base class tester and input data
  TestAC(int argc, char *argv[], const bool silent,
         const std::string &name, const std::vector<std::string> &options,
         const Routine clblast_lambda, const Routine clblas_lambda);

  // The test functions, taking no inputs
  void TestRegular(Arguments<T> &args, const std::string &name);
  void TestInvalidBufferSizes(Arguments<T> &args, const std::string &name);

 private:

  // Source data to test with
  std::vector<T> a_source_;
  std::vector<T> c_source_;
  
  // The routines to test
  Routine clblast_lambda_;
  Routine clblas_lambda_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_CORRECTNESS_TESTAC_H_
#endif
