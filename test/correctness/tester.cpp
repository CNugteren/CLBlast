
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
#include <cstdio>
#include <cstdlib>

#include "test/correctness/tester.hpp"

namespace clblast {
// =================================================================================================

// Relative error margins
template <typename T>
float getRelativeErrorMargin() {
  return 0.005f; // 0.5% is considered acceptable for float/double-precision
}
template float getRelativeErrorMargin<float>(); // as the above default
template float getRelativeErrorMargin<double>(); // as the above default
template float getRelativeErrorMargin<float2>(); // as the above default
template float getRelativeErrorMargin<double2>(); // as the above default
template <>
float getRelativeErrorMargin<half>() {
  return 0.080f; // 8% (!) error is considered acceptable for half-precision
}

// Absolute error margins
template <typename T>
float getAbsoluteErrorMargin() {
  return 0.001f;
}
template float getAbsoluteErrorMargin<float>(); // as the above default
template float getAbsoluteErrorMargin<double>(); // as the above default
template float getAbsoluteErrorMargin<float2>(); // as the above default
template float getAbsoluteErrorMargin<double2>(); // as the above default
template <>
float getAbsoluteErrorMargin<half>() {
  return 0.15f; // especially small values are inaccurate for half-precision
}

// L2 error margins
template <typename T>
double getL2ErrorMargin() {
  return 0.0f; // zero means don't look at the L2 error margin at all, use the other metrics
}
template double getL2ErrorMargin<float>(); // as the above default
template double getL2ErrorMargin<double>(); // as the above default
template double getL2ErrorMargin<float2>(); // as the above default
template double getL2ErrorMargin<double2>(); // as the above default
template <>
double getL2ErrorMargin<half>() {
  return 0.05; // half-precision results are considered OK as long as the L2 error is low enough
}

// Error margin: numbers beyond this value are considered equal to inf or NaN
template <typename T>
T getAlmostInfNumber() {
  return static_cast<T>(1e35); // used for correctness testing of TRSV and TRSM routines
}

// =================================================================================================

// General constructor for all CLBlast testers. It prints out the test header to stdout and sets-up
// the clBLAS library for reference.
template <typename T, typename U>
Tester<T,U>::Tester(const std::vector<std::string> &arguments, const bool silent,
                    const std::string &name, const std::vector<std::string> &options):
    help_("Options given/available:\n"),
    platform_(Platform(GetArgument(arguments, help_, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0})))),
    device_(Device(platform_, GetArgument(arguments, help_, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0})))),
    context_(Context(device_)),
    queue_(Queue(context_, device_)),
    full_test_(CheckArgument(arguments, help_, kArgFullTest)),
    verbose_(CheckArgument(arguments, help_, kArgVerbose)),
    error_log_{},
    num_passed_{0},
    num_skipped_{0},
    num_failed_{0},
    print_count_{0},
    tests_passed_{0},
    tests_skipped_{0},
    tests_failed_{0} {
  options_ = options;

  // Determines which reference is the default
  #if defined(CLBLAST_REF_CBLAS)
      auto default_cblas = 0;
  #endif
  #if defined(CLBLAST_REF_CLBLAS)
      auto default_clblas = 0;
  #endif
  #if defined(CLBLAST_REF_CUBLAS)
      auto default_cublas = 0;
  #endif
  #if defined(CLBLAST_REF_CBLAS)
    default_cblas = 1;
  #elif defined(CLBLAST_REF_CLBLAS)
    default_clblas = 1;
  #elif defined(CLBLAST_REF_CUBLAS)
    default_cublas = 1;
  #endif

  // Determines which reference to test against
  compare_clblas_ = 0;
  compare_cblas_ = 0;
  compare_cublas_ = 0;
  #if defined(CLBLAST_REF_CBLAS)
    compare_cblas_  = GetArgument(arguments, help_, kArgComparecblas, default_cblas);
  #endif
  #if defined(CLBLAST_REF_CLBLAS)
    compare_clblas_ = GetArgument(arguments, help_, kArgCompareclblas, default_clblas);
  #endif
  #if defined(CLBLAST_REF_CUBLAS)
    compare_cublas_  = GetArgument(arguments, help_, kArgComparecublas, default_cublas);
  #endif

  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help_.c_str()); }

  // Support for cuBLAS not available yet
  if (compare_cublas_) { throw std::runtime_error("Cannot test against cuBLAS; not implemented yet"); }

  // Can only test against a single reference (not two, not zero)
  if (compare_clblas_ && compare_cblas_) {
    throw std::runtime_error("Cannot test against both clBLAS and CBLAS references; choose one using the -cblas and -clblas arguments");
  }
  if (!compare_clblas_ && !compare_cblas_) {
    throw std::runtime_error("Choose one reference (clBLAS or CBLAS) to test against using the -cblas and -clblas arguments");
  }

  // Prints the header
  fprintf(stdout, "* Running on OpenCL device '%s'.\n", GetDeviceName(device_).c_str());
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
  fprintf(stdout, "* Testing with error margins of %.1lf%% (relative) and %.3lf (absolute)\n",
          100.0f * getRelativeErrorMargin<T>(), getAbsoluteErrorMargin<T>());
  if (getL2ErrorMargin<T>() != 0.0f) {
    fprintf(stdout, "* and a combined maximum allowed L2 error of %.2e\n", getL2ErrorMargin<T>());
  }

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
    std::cout << "* Completed all test-cases for this routine. Results:" << std::endl;
    std::cout << "   " << tests_passed_ << " test(s) passed" << std::endl;
    if (tests_skipped_ > 0) { std::cout << kPrintWarning; }
    std::cout << "   " << tests_skipped_ << " test(s) skipped" << kPrintEnd << std::endl;
    if (tests_failed_ > 0) { std::cout << kPrintError; }
    std::cout << "   " << tests_failed_ << " test(s) failed" << kPrintEnd << std::endl;
  }
  std::cout << std::endl;

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
  std::cout << " " << num_passed_ << " passed /";
  if (num_skipped_ != 0) {
    std::cout << " " << kPrintWarning << num_skipped_ << " skipped" << kPrintEnd << " /";
  }
  else {
    std::cout << " " << num_skipped_ << " skipped /";
  }
  if (num_failed_ != 0) {
    std::cout << " " << kPrintError << num_failed_ << " failed" << kPrintEnd << std::endl;
  }
  else {
    std::cout << " " << num_failed_ << " failed" << std::endl;
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

  // Either an OpenCL or CLBlast internal error occurred, fail the test immediately
  // NOTE: the OpenCL error codes grow downwards without any declared lower bound, hence the magic
  // number. The last error code is atm around -70, but -500 is chosen to be on the safe side.
  if (clblast_status != StatusCode::kSuccess &&
      (clblast_status > static_cast<StatusCode>(-500) /* matches OpenCL errors (see above) */ ||
       clblast_status < StatusCode::kNotImplemented) /* matches CLBlast internal errors */) {
    PrintTestResult(kErrorStatus);
    ReportError({StatusCode::kSuccess, clblast_status, kStatusError, args});
    if (verbose_) {
      fprintf(stdout, "\n");
      PrintErrorLog({{StatusCode::kSuccess, clblast_status, kStatusError, args}});
      fprintf(stdout, "   ");
    }
  }

  // Routine is not implemented
  else if (clblast_status == StatusCode::kNotImplemented) {
    PrintTestResult(kSkippedCompilation);
    ReportSkipped();
  }

  // Cannot compare error codes against a library other than clBLAS
  else if (compare_cblas_) {
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
    if (o == kArgAlpha)    { result += kArgAlpha + equals + ToString(args.alpha) + " "; }
    if (o == kArgBeta)     { result += kArgBeta + equals + ToString(args.beta) + " "; }
    if (o == kArgBatchCount){result += kArgBatchCount + equals + ToString(args.batch_count) + " "; }
    if (o == kArgKernelMode){result += kArgKernelMode + equals + ToString(args.kernel_mode) + " "; }
    if (o == kArgChannels) { result += kArgChannels + equals + ToString(args.channels) + " "; }
    if (o == kArgHeight)   { result += kArgHeight + equals + ToString(args.height) + " "; }
    if (o == kArgWidth)    { result += kArgWidth + equals + ToString(args.width) + " "; }
    if (o == kArgNumKernels){result += kArgNumKernels + equals + ToString(args.num_kernels) + " "; }
    if (o == kArgKernelH)  { result += kArgKernelH + equals + ToString(args.kernel_h) + " "; }
    if (o == kArgKernelW)  { result += kArgKernelW + equals + ToString(args.kernel_w) + " "; }
    if (o == kArgPadH)     { result += kArgPadH + equals + ToString(args.pad_h) + " "; }
    if (o == kArgPadW)     { result += kArgPadW + equals + ToString(args.pad_w) + " "; }
    if (o == kArgStrideH)  { result += kArgStrideH + equals + ToString(args.stride_h) + " "; }
    if (o == kArgStrideW)  { result += kArgStrideW + equals + ToString(args.stride_w) + " "; }
    if (o == kArgDilationH){ result += kArgDilationH + equals + ToString(args.dilation_h) + " "; }
    if (o == kArgDilationW){ result += kArgDilationW + equals + ToString(args.dilation_w) + " "; }
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
      fprintf(stdout, "   Error rate %.2lf%%: ", entry.error_percentage);
    }
    else {
      fprintf(stdout, "   Status code %d (expected %d): ",
              static_cast<int>(entry.status_found),
              static_cast<int>(entry.status_expect));
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
  // Handles cases with both results NaN or inf
  else if ((std::isnan(val1) && std::isnan(val2)) || (std::isinf(val1) && std::isinf(val2))) {
    return true;
  }
  // Also considers it OK if one of the results in NaN and the other is inf
  // Note: for TRSV and TRSM routines
  else if ((std::isnan(val1) && std::isinf(val2)) || (std::isinf(val1) && std::isnan(val2))) {
    return true;
  }
  // Also considers it OK if one of the values is super large and the other is inf or NaN
  // Note: for TRSV and TRSM routines
  else if ((std::abs(val1) > getAlmostInfNumber<T>() && (std::isinf(val2) || std::isnan(val2))) ||
           (std::abs(val2) > getAlmostInfNumber<T>() && (std::isinf(val1) || std::isnan(val1)))) {
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
  const auto kErrorMarginRelative = static_cast<T>(getRelativeErrorMargin<T>());
  const auto kErrorMarginAbsolute = static_cast<T>(getAbsoluteErrorMargin<T>());
  return TestSimilarityNear(val1, val2, kErrorMarginAbsolute, kErrorMarginRelative);
}

// Compiles the default case for standard data-types
template bool TestSimilarity<float>(const float, const float);
template bool TestSimilarity<double>(const double, const double);

// Specialisations for non-standard data-types
template <>
bool TestSimilarity(const float2 val1, const float2 val2) {
  const auto real = TestSimilarity(val1.real(), val2.real());
  const auto imag = TestSimilarity(val1.imag(), val2.imag());
  if (real && imag) { return true; }
  // also OK if one is good and the combined is good (indicates a big diff between real & imag)
  if (real || imag) { return TestSimilarity(val1.real() + val1.imag(), val2.real() + val2.imag()); }
  return false; // neither real nor imag is good, return false
}
template <>
bool TestSimilarity(const double2 val1, const double2 val2) {
  const auto real = TestSimilarity(val1.real(), val2.real());
  const auto imag = TestSimilarity(val1.imag(), val2.imag());
  if (real && imag) { return true; }
  // also OK if one is good and the combined is good (indicates a big diff between real & imag)
  if (real || imag) { return TestSimilarity(val1.real() + val1.imag(), val2.real() + val2.imag()); }
  return false; // neither real nor imag is good, return false
}
template <>
bool TestSimilarity(const half val1, const half val2) {
  const auto kErrorMarginRelative = getRelativeErrorMargin<half>();
  const auto kErrorMarginAbsolute = getAbsoluteErrorMargin<half>();
  return TestSimilarityNear(HalfToFloat(val1), HalfToFloat(val2),
                            kErrorMarginAbsolute, kErrorMarginRelative);
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
