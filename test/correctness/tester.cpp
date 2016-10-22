
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

#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "test/correctness/tester.hpp"

namespace clblast {
// =================================================================================================

// Maximum number of test results printed on a single line
template <typename T, typename U> const size_t Tester<T,U>::kResultsPerLine = size_t{64};

// Error percentage is not applicable: error was caused by an incorrect status
template <typename T, typename U> const float Tester<T,U>::kStatusError = -1.0f;

// Constants holding start and end strings for terminal-output in colour
template <typename T, typename U> const std::string Tester<T,U>::kPrintError = "\x1b[31m";
template <typename T, typename U> const std::string Tester<T,U>::kPrintSuccess = "\x1b[32m";
template <typename T, typename U> const std::string Tester<T,U>::kPrintWarning = "\x1b[35m";
template <typename T, typename U> const std::string Tester<T,U>::kPrintMessage = "\x1b[1m";
template <typename T, typename U> const std::string Tester<T,U>::kPrintEnd = "\x1b[0m";

// Sets the output error coding
template <typename T, typename U> const std::string Tester<T,U>::kSuccessData = kPrintSuccess + ":" + kPrintEnd;
template <typename T, typename U> const std::string Tester<T,U>::kSuccessStatus = kPrintSuccess + "." + kPrintEnd;
template <typename T, typename U> const std::string Tester<T,U>::kErrorData = kPrintError + "X" + kPrintEnd;
template <typename T, typename U> const std::string Tester<T,U>::kErrorStatus = kPrintError + "/" + kPrintEnd;
template <typename T, typename U> const std::string Tester<T,U>::kSkippedCompilation = kPrintWarning + "\\" + kPrintEnd;
template <typename T, typename U> const std::string Tester<T,U>::kUnsupportedPrecision = kPrintWarning + "o" + kPrintEnd;
template <typename T, typename U> const std::string Tester<T,U>::kUnsupportedReference = kPrintWarning + "-" + kPrintEnd;

// =================================================================================================

// General constructor for all CLBlast testers. It prints out the test header to stdout and sets-up
// the clBLAS library for reference.
template <typename T, typename U>
Tester<T,U>::Tester(int argc, char *argv[], const bool silent,
                    const std::string &name, const std::vector<std::string> &options):
    help_("Options given/available:\n"),
    platform_(Platform(GetArgument(argc, argv, help_, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0})))),
    device_(Device(platform_, GetArgument(argc, argv, help_, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0})))),
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
    tests_failed_{0} {
  options_ = options;

  // Determines which reference to test against
  #if defined(CLBLAST_REF_CLBLAS) && defined(CLBLAST_REF_CBLAS)
    compare_clblas_ = GetArgument(argc, argv, help_, kArgCompareclblas, 0);
    compare_cblas_  = GetArgument(argc, argv, help_, kArgComparecblas, 1);
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
  if (!verbose_) { fprintf(stdout, "   "); }

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
  if (!verbose_) { fprintf(stdout, "\n"); }
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
  else if (clblast_status == StatusCode::kOpenCLBuildProgramFailure ||
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

// Retrieves the options as a string for a specific test
template <typename T, typename U>
std::string Tester<T,U>::GetOptionsString(const Arguments<U> &args) {
  auto result = std::string("");
  const auto equals = std::string("=");
  for (auto &o: options_) {
    if (o == kArgM)        { result += kArgM + equals + ToString(args.m) + " "; }
    if (o == kArgN)        { result += kArgN + equals + ToString(args.n) + " "; }
    if (o == kArgK)        { result += kArgK + equals + ToString(args.k) + " "; }
    if (o == kArgKU)       { result += kArgKU + equals + ToString(args.ku) + " "; }
    if (o == kArgKL)       { result += kArgKL + equals + ToString(args.kl) + " "; }
    if (o == kArgXInc)     { result += kArgXInc + equals + ToString(args.x_inc) + " "; }
    if (o == kArgYInc)     { result += kArgYInc + equals + ToString(args.y_inc) + " "; }
    if (o == kArgXOffset)  { result += kArgXOffset + equals + ToString(args.x_offset) + " "; }
    if (o == kArgYOffset)  { result += kArgYOffset + equals + ToString(args.y_offset) + " "; }
    if (o == kArgALeadDim) { result += kArgALeadDim + equals + ToString(args.a_ld) + " "; }
    if (o == kArgBLeadDim) { result += kArgBLeadDim + equals + ToString(args.b_ld) + " "; }
    if (o == kArgCLeadDim) { result += kArgCLeadDim + equals + ToString(args.c_ld) + " "; }
    if (o == kArgAOffset)  { result += kArgAOffset + equals + ToString(args.a_offset) + " "; }
    if (o == kArgBOffset)  { result += kArgBOffset + equals + ToString(args.b_offset) + " "; }
    if (o == kArgCOffset)  { result += kArgCOffset + equals + ToString(args.c_offset) + " "; }
    if (o == kArgAPOffset) { result += kArgAPOffset + equals + ToString(args.ap_offset) + " "; }
    if (o == kArgDotOffset){ result += kArgDotOffset + equals + ToString(args.dot_offset) + " "; }
  }
  return result;
}

// As above, but now only prints information relevant to invalid buffer sizes
template <typename T, typename U>
std::string Tester<T,U>::GetSizesString(const Arguments<U> &args) {
  auto result = std::string("");
  const auto equals = std::string("=");
  for (auto &o: options_) {
    if (o == kArgM)        { result += kArgM + equals + ToString(args.m) + " "; }
    if (o == kArgN)        { result += kArgN + equals + ToString(args.n) + " "; }
    if (o == kArgK)        { result += kArgK + equals + ToString(args.k) + " "; }
    if (o == kArgXOffset)  { result += "xsize" + equals + ToString(args.x_size) + " "; }
    if (o == kArgYOffset)  { result += "ysize" + equals + ToString(args.y_size) + " "; }
    if (o == kArgAOffset)  { result += "asize" + equals + ToString(args.a_size) + " "; }
    if (o == kArgBOffset)  { result += "bsize" + equals + ToString(args.b_size) + " "; }
    if (o == kArgCOffset)  { result += "csize" + equals + ToString(args.c_size) + " "; }
    if (o == kArgAPOffset) { result += "apsize" + equals + ToString(args.ap_size) + " "; }
    if (o == kArgDotOffset){ result += "scalarsize" + equals + ToString(args.scalar_size) + " "; }
  }
  return result;
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
  if (verbose_) {
    fprintf(stdout, "%s\n", message.c_str());
  }
  else
  {
    if (print_count_ == kResultsPerLine) {
      print_count_ = 0;
      fprintf(stdout, "\n   ");
    }
    fprintf(stdout, "%s", message.c_str());
    print_count_++;
  }
  std::cout << std::flush;
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
    fprintf(stdout, "%s\n", GetOptionsString(entry.args).c_str());
  }
}

// =================================================================================================
// Below are the non-member functions (separated because of otherwise required partial class
// template specialization)
// =================================================================================================

// Compares two floating point values and returns whether they are within an acceptable error
// margin. This replaces GTest's EXPECT_NEAR().
template <typename T>
bool TestSimilarityNear(const T val1, const T val2,
                        const T error_margin_absolute, const T error_margin_relative) {
  const auto difference = std::fabs(val1 - val2);

  // Shortcut, handles infinities
  if (val1 == val2) {
    return true;
  }
  // The values are zero or very small: the relative error is less meaningful
  else if (val1 == 0 || val2 == 0 || difference < error_margin_absolute) {
    return (difference < error_margin_absolute);
  }
  // Use relative error
  else {
    const auto absolute_sum = std::fabs(val1) + std::fabs(val2);
    return (difference / absolute_sum) < error_margin_relative;
  }
}

// Default method for similarity testing
template <typename T>
bool TestSimilarity(const T val1, const T val2) {
  constexpr auto kErrorMarginRelative = T(0.025);
  constexpr auto kErrorMarginAbsolute = T(0.001);
  return TestSimilarityNear(val1, val2, kErrorMarginRelative, kErrorMarginAbsolute);
}

// Compiles the default case for standard data-types
template bool TestSimilarity<float>(const float, const float);
template bool TestSimilarity<double>(const double, const double);

// Specialisations for non-standard data-types
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
template <>
bool TestSimilarity(const half val1, const half val2) {
  constexpr auto kErrorMarginRelative = float(0.050);
  constexpr auto kErrorMarginAbsolute = float(0.002);
  return TestSimilarityNear(HalfToFloat(val1), HalfToFloat(val2),
                            kErrorMarginRelative, kErrorMarginAbsolute);
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
template <> const std::vector<half> GetExampleScalars(const bool full_test) {
  if (full_test) { return {FloatToHalf(0.0f), FloatToHalf(1.0f), FloatToHalf(3.14f)}; }
  else { return {FloatToHalf(3.14f)}; }
}

// =================================================================================================

// Compiles the templated class
template class Tester<half, half>;
template class Tester<float, float>;
template class Tester<double, double>;
template class Tester<float2, float2>;
template class Tester<double2, double2>;
template class Tester<float2, float>;
template class Tester<double2, double>;

// =================================================================================================
} // namespace clblast
