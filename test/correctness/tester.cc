
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Tester class (see the header for information about the class).
//
// =================================================================================================

#include "correctness/tester.h"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

namespace clblast {
// =================================================================================================

// General constructor for all CLBlast testers. It prints out the test header to stdout and sets-up
// the clBLAS library for reference.
template <typename T, typename U>
Tester<T,U>::Tester(int argc, char *argv[], const bool silent,
                    const std::string &name, const std::vector<std::string> &options):
    help_("Options given/available:\n"),
    platform_(Platform(GetArgument(argc, argv, help_, kArgPlatform, size_t{0}))),
    device_(Device(platform_, GetArgument(argc, argv, help_, kArgDevice, size_t{0}))),
    context_(Context(device_)),
    queue_(Queue(context_, device_)),
    full_test_(CheckArgument(argc, argv, help_, kArgFullTest)),
    verbose_(CheckArgument(argc, argv, help_, kArgVerbose)),
    error_log_{},
    num_passed_{0},
    num_skipped_{0},
    num_failed_{0},
    print_count_{0},
    tests_passed_{0},
    tests_skipped_{0},
    tests_failed_{0},
    options_{options} {

  // Determines which reference to test against
  #if defined(CLBLAST_REF_CLBLAS) && defined(CLBLAST_REF_CBLAS)
    compare_clblas_ = GetArgument(argc, argv, help_, kArgCompareclblas, 1);
    compare_cblas_  = GetArgument(argc, argv, help_, kArgComparecblas, 0);
  #elif CLBLAST_REF_CLBLAS
    compare_clblas_ = GetArgument(argc, argv, help_, kArgCompareclblas, 1);
    compare_cblas_ = 0;
  #elif CLBLAST_REF_CBLAS
    compare_clblas_ = 0;
    compare_cblas_  = GetArgument(argc, argv, help_, kArgComparecblas, 1);
  #else
    compare_clblas_ = 0;
    compare_cblas_ = 0;
  #endif

  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help_.c_str()); }

  // Can only test against a single reference (not two, not zero)
  if (compare_clblas_ && compare_cblas_) {
    throw std::runtime_error("Cannot test against both clBLAS and CBLAS references; choose one using the -cblas and -clblas arguments");
  }
  if (!compare_clblas_ && !compare_cblas_) {
    throw std::runtime_error("Choose one reference (clBLAS or CBLAS) to test against using the -cblas and -clblas arguments");
  }

  // Prints the header
  fprintf(stdout, "* Running on OpenCL device '%s'.\n", device_.Name().c_str());
  fprintf(stdout, "* Starting tests for the %s'%s'%s routine.",
          kPrintMessage.c_str(), name.c_str(), kPrintEnd.c_str());

  // Checks whether the precision is supported
  if (!PrecisionSupported<T>(device_)) {
    fprintf(stdout, "\n* All tests skipped: %sUnsupported precision%s\n",
            kPrintWarning.c_str(), kPrintEnd.c_str());
    return;
  }

  // Prints the legend
  fprintf(stdout, " Legend:\n");
  fprintf(stdout, "   %s -> Test produced correct results\n", kSuccessData.c_str());
  fprintf(stdout, "   %s -> Test returned the correct error code\n", kSuccessStatus.c_str());
  fprintf(stdout, "   %s -> Test produced incorrect results\n", kErrorData.c_str());
  fprintf(stdout, "   %s -> Test returned an incorrect error code\n", kErrorStatus.c_str());
  fprintf(stdout, "   %s -> Test not executed: OpenCL-kernel compilation error\n",
          kSkippedCompilation.c_str());
  fprintf(stdout, "   %s -> Test not executed: Unsupported precision\n",
          kUnsupportedPrecision.c_str());
  fprintf(stdout, "   %s -> Test not completed: Reference CBLAS doesn't output error codes\n",
          kUnsupportedReference.c_str());

  // Initializes clBLAS
  #ifdef CLBLAST_REF_CLBLAS
    if (compare_clblas_) {
      auto status = clblasSetup();
      if (status != CL_SUCCESS) {
        throw std::runtime_error("clBLAS setup error: "+ToString(static_cast<int>(status)));
      }
    }
  #endif
}

// Destructor prints the summary of the test cases and cleans-up the clBLAS library
template <typename T, typename U>
Tester<T,U>::~Tester() {
  if (PrecisionSupported<T>(device_)) {
    fprintf(stdout, "* Completed all test-cases for this routine. Results:\n");
    fprintf(stdout, "   %zu test(s) passed\n", tests_passed_);
    if (tests_skipped_ > 0) { fprintf(stdout, "%s", kPrintWarning.c_str()); }
    fprintf(stdout, "   %zu test(s) skipped%s\n", tests_skipped_, kPrintEnd.c_str());
    if (tests_failed_ > 0) { fprintf(stdout, "%s", kPrintError.c_str()); }
    fprintf(stdout, "   %zu test(s) failed%s\n", tests_failed_, kPrintEnd.c_str());
  }
  fprintf(stdout, "\n");

  // Cleans-up clBLAS
  #ifdef CLBLAST_REF_CLBLAS
    if (compare_clblas_) {
      clblasTeardown();
    }
  #endif
}

// =================================================================================================

// Function called at the start of each test. This prints a header with information about the
// test and re-initializes all test data-structures.
template <typename T, typename U>
void Tester<T,U>::TestStart(const std::string &test_name, const std::string &test_configuration) {

  // Prints the header
  fprintf(stdout, "* Testing %s'%s'%s for %s'%s'%s:\n",
          kPrintMessage.c_str(), test_name.c_str(), kPrintEnd.c_str(),
          kPrintMessage.c_str(), test_configuration.c_str(), kPrintEnd.c_str());
  fprintf(stdout, "   ");

  // Empties the error log and the error/pass counters
  error_log_.clear();
  num_passed_ = 0;
  num_skipped_ = 0;
  num_failed_ = 0;
  print_count_ = 0;
}

// Function called at the end of each test. This prints errors if any occured. It also prints a
// summary of the number of sub-tests passed/failed.
template <typename T, typename U>
void Tester<T,U>::TestEnd() {
  fprintf(stdout, "\n");
  tests_passed_ += num_passed_;
  tests_skipped_ += num_skipped_;
  tests_failed_ += num_failed_;

  // Prints the errors
  PrintErrorLog(error_log_);

  // Prints a test summary
  auto pass_rate = 100*num_passed_ / static_cast<float>(num_passed_ + num_skipped_ + num_failed_);
  fprintf(stdout, "   Pass rate %s%5.1lf%%%s:", kPrintMessage.c_str(), pass_rate, kPrintEnd.c_str());
  fprintf(stdout, " %zu passed /", num_passed_);
  if (num_skipped_ != 0) {
    fprintf(stdout, " %s%zu skipped%s /", kPrintWarning.c_str(), num_skipped_, kPrintEnd.c_str());
  }
  else {
    fprintf(stdout, " %zu skipped /", num_skipped_);
  }
  if (num_failed_ != 0) {
    fprintf(stdout, " %s%zu failed%s\n", kPrintError.c_str(), num_failed_, kPrintEnd.c_str());
  }
  else {
    fprintf(stdout, " %zu failed\n", num_failed_);
  }
}

// =================================================================================================

// Handles a 'pass' or 'error' depending on whether there are any errors
template <typename T, typename U>
void Tester<T,U>::TestErrorCount(const size_t errors, const size_t size, const Arguments<U> &args) {

  // Finished successfully
  if (errors == 0) {
    PrintTestResult(kSuccessData);
    ReportPass();
  }

  // Error(s) occurred
  else {
    auto percentage = 100*errors / static_cast<float>(size);
    PrintTestResult(kErrorData);
    ReportError({StatusCode::kSuccess, StatusCode::kSuccess, percentage, args});
  }
}

// Compares two status codes for equality. The outcome can be a pass (they are the same), a warning
// (CLBlast reported a compilation error), or an error (they are different).
template <typename T, typename U>
void Tester<T,U>::TestErrorCodes(const StatusCode clblas_status, const StatusCode clblast_status,
                                 const Arguments<U> &args) {

  // Cannot compare error codes against a library other than clBLAS
  if (compare_cblas_) {
    PrintTestResult(kUnsupportedReference);
    ReportSkipped();
  }

  // Finished successfully
  else if (clblas_status == clblast_status) {
    PrintTestResult(kSuccessStatus);
    ReportPass();
  }

  // No support for this kind of precision
  else if (clblast_status == StatusCode::kNoDoublePrecision ||
           clblast_status == StatusCode::kNoHalfPrecision) {
    PrintTestResult(kUnsupportedPrecision);
    ReportSkipped();
  }

  // Could not compile the CLBlast kernel properly
  else if (clblast_status == StatusCode::kBuildProgramFailure ||
           clblast_status == StatusCode::kNotImplemented) {
    PrintTestResult(kSkippedCompilation);
    ReportSkipped();
  }

  // Error occurred
  else {
    PrintTestResult(kErrorStatus);
    ReportError({clblas_status, clblast_status, kStatusError, args});
    if (verbose_) {
      fprintf(stdout, "\n");
      PrintErrorLog({{clblas_status, clblast_status, kStatusError, args}});
      fprintf(stdout, "   ");
    }
  }
}

// =================================================================================================

// Retrieves the offset values to test with
template <typename T, typename U>
const std::vector<size_t> Tester<T,U>::GetOffsets() const {
  if (full_test_) { return {0, 10}; }
  else { return {0}; }
}

// =================================================================================================

// A test can either pass, be skipped, or fail
template <typename T, typename U>
void Tester<T,U>::ReportPass() {
  num_passed_++;
}
template <typename T, typename U>
void Tester<T,U>::ReportSkipped() {
  num_skipped_++;
}
template <typename T, typename U>
void Tester<T,U>::ReportError(const ErrorLogEntry &error_log_entry) {
  error_log_.push_back(error_log_entry);
  num_failed_++;
}

// =================================================================================================

// Prints the test-result symbol to screen. This function limits the maximum number of symbols per
// line by printing newlines once every so many calls.
template <typename T, typename U>
void Tester<T,U>::PrintTestResult(const std::string &message) {
  if (print_count_ == kResultsPerLine) {
    print_count_ = 0;
    fprintf(stdout, "\n   ");
  }
  fprintf(stdout, "%s", message.c_str());
  std::cout << std::flush;
  print_count_++;
}

// Prints details of errors occurred in a given error log
template <typename T, typename U>
void Tester<T,U>::PrintErrorLog(const std::vector<ErrorLogEntry> &error_log) {
  for (auto &entry: error_log) {
    if (entry.error_percentage != kStatusError) {
      fprintf(stdout, "   Error rate %.1lf%%: ", entry.error_percentage);
    }
    else {
      fprintf(stdout, "   Status code %d (expected %d): ", entry.status_found, entry.status_expect);
    }
    for (auto &o: options_) {
      if (o == kArgM)        { fprintf(stdout, "%s=%zu ", kArgM, entry.args.m); }
      if (o == kArgN)        { fprintf(stdout, "%s=%zu ", kArgN, entry.args.n); }
      if (o == kArgK)        { fprintf(stdout, "%s=%zu ", kArgK, entry.args.k); }
      if (o == kArgKU)       { fprintf(stdout, "%s=%zu ", kArgKU, entry.args.ku); }
      if (o == kArgKL)       { fprintf(stdout, "%s=%zu ", kArgKL, entry.args.kl); }
      if (o == kArgLayout)   { fprintf(stdout, "%s=%d ", kArgLayout, entry.args.layout);}
      if (o == kArgATransp)  { fprintf(stdout, "%s=%d ", kArgATransp, entry.args.a_transpose);}
      if (o == kArgBTransp)  { fprintf(stdout, "%s=%d ", kArgBTransp, entry.args.b_transpose);}
      if (o == kArgSide)     { fprintf(stdout, "%s=%d ", kArgSide, entry.args.side);}
      if (o == kArgTriangle) { fprintf(stdout, "%s=%d ", kArgTriangle, entry.args.triangle);}
      if (o == kArgDiagonal) { fprintf(stdout, "%s=%d ", kArgDiagonal, entry.args.diagonal);}
      if (o == kArgXInc)     { fprintf(stdout, "%s=%zu ", kArgXInc, entry.args.x_inc);}
      if (o == kArgYInc)     { fprintf(stdout, "%s=%zu ", kArgYInc, entry.args.y_inc);}
      if (o == kArgXOffset)  { fprintf(stdout, "%s=%zu ", kArgXOffset, entry.args.x_offset);}
      if (o == kArgYOffset)  { fprintf(stdout, "%s=%zu ", kArgYOffset, entry.args.y_offset);}
      if (o == kArgALeadDim) { fprintf(stdout, "%s=%zu ", kArgALeadDim, entry.args.a_ld);}
      if (o == kArgBLeadDim) { fprintf(stdout, "%s=%zu ", kArgBLeadDim, entry.args.b_ld);}
      if (o == kArgCLeadDim) { fprintf(stdout, "%s=%zu ", kArgCLeadDim, entry.args.c_ld);}
      if (o == kArgAOffset)  { fprintf(stdout, "%s=%zu ", kArgAOffset, entry.args.a_offset);}
      if (o == kArgBOffset)  { fprintf(stdout, "%s=%zu ", kArgBOffset, entry.args.b_offset);}
      if (o == kArgCOffset)  { fprintf(stdout, "%s=%zu ", kArgCOffset, entry.args.c_offset);}
      if (o == kArgAPOffset) { fprintf(stdout, "%s=%zu ", kArgAPOffset, entry.args.ap_offset);}
      if (o == kArgDotOffset){ fprintf(stdout, "%s=%zu ", kArgDotOffset, entry.args.dot_offset);}
    }
    fprintf(stdout, "\n");
  }
}

// =================================================================================================
// Below are the non-member functions (separated because of otherwise required partial class
// template specialization)
// =================================================================================================

// Compares two floating point values and returns whether they are within an acceptable error
// margin. This replaces GTest's EXPECT_NEAR().
template <typename T>
bool TestSimilarity(const T val1, const T val2) {
  const auto difference = std::fabs(val1 - val2);

  // Set the allowed error margin for floating-point comparisons
  constexpr auto kErrorMarginRelative = T{0.025};
  constexpr auto kErrorMarginAbsolute = T{1.0e-4};

  // Shortcut, handles infinities
  if (val1 == val2) {
    return true;
  }
  // The values are zero or very small: the relative error is less meaningful
  else if (val1 == 0 || val2 == 0 || difference < kErrorMarginAbsolute) {
    return (difference < kErrorMarginAbsolute);
  }
  // Use relative error
  else {
    const auto absolute_sum = std::fabs(val1) + std::fabs(val2);
    return (difference / absolute_sum) < kErrorMarginRelative;
  }
}

// Compiles the default case for non-complex data-types
template bool TestSimilarity<float>(const float, const float);
template bool TestSimilarity<double>(const double, const double);

// Specialisations for complex data-types
template <>
bool TestSimilarity(const float2 val1, const float2 val2) {
  auto real = TestSimilarity(val1.real(), val2.real());
  auto imag = TestSimilarity(val1.imag(), val2.imag());
  return (real && imag);
}
template <>
bool TestSimilarity(const double2 val1, const double2 val2) {
  auto real = TestSimilarity(val1.real(), val2.real());
  auto imag = TestSimilarity(val1.imag(), val2.imag());
  return (real && imag);
}

// =================================================================================================

// Retrieves a list of example scalar values, used for the alpha and beta arguments for the various
// routines. This function is specialised for the different data-types.
template <> const std::vector<float> GetExampleScalars(const bool full_test) {
  if (full_test) { return {0.0f, 1.0f, 3.14f}; }
  else { return {3.14f}; }
}
template <> const std::vector<double> GetExampleScalars(const bool full_test) {
  if (full_test) { return {0.0, 1.0, 3.14}; }
  else { return {3.14}; }
}
template <> const std::vector<float2> GetExampleScalars(const bool full_test) {
  if (full_test) { return {{0.0f, 0.0f}, {1.0f, 1.3f}, {2.42f, 3.14f}}; }
  else { return {{2.42f, 3.14f}}; }
}
template <> const std::vector<double2> GetExampleScalars(const bool full_test) {
  if (full_test) { return {{0.0, 0.0}, {1.0, 1.3}, {2.42, 3.14}}; }
  else { return {{2.42, 3.14}}; }
}

// =================================================================================================

// Compiles the templated class
template class Tester<float, float>;
template class Tester<double, double>;
template class Tester<float2, float2>;
template class Tester<double2, double2>;
template class Tester<float2, float>;
template class Tester<double2, double>;

// =================================================================================================
} // namespace clblast
