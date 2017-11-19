
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Tester class, providing a test-framework. GTest was used before, but
// was not able to handle certain cases (e.g. template type + parameters). This is its (basic)
// custom replacement.
// Typename T: the data-type of the routine's memory buffers (==precision)
// Typename U: the data-type of the alpha and beta arguments
//
// =================================================================================================

#ifndef CLBLAST_TEST_CORRECTNESS_TESTER_H_
#define CLBLAST_TEST_CORRECTNESS_TESTER_H_

#include <string>
#include <vector>
#include <memory>

#include "utilities/utilities.hpp"
#include "test/test_utilities.hpp"

// The libraries
#ifdef CLBLAST_REF_CLBLAS
  #include <clBLAS.h>
#endif

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Tester {
 public:

  // Maximum number of test results printed on a single line
  static const size_t kResultsPerLine;

  // Error percentage is not applicable: error was caused by an incorrect status
  static const float kStatusError;

  // Constants holding start and end strings for terminal-output in colour
  static const std::string kPrintError;
  static const std::string kPrintSuccess;
  static const std::string kPrintWarning;
  static const std::string kPrintMessage;
  static const std::string kPrintEnd;

  // Sets the output error coding
  static const std::string kSuccessData;
  static const std::string kSuccessStatus;
  static const std::string kErrorData;
  static const std::string kErrorStatus;
  static const std::string kSkippedCompilation;
  static const std::string kUnsupportedPrecision;
  static const std::string kUnsupportedReference;

  // This structure combines the above log-entry with a status code an error percentage
  struct ErrorLogEntry {
    StatusCode status_expect;
    StatusCode status_found;
    float error_percentage;
    Arguments<U> args;
  };

  // Creates an instance of the tester, running on a particular OpenCL platform and device. It
  // takes the routine's names as an additional parameter.
  explicit Tester(const std::vector<std::string> &arguments, const bool silent,
                  const std::string &name, const std::vector<std::string> &options);
  ~Tester();

  // These methods start and end a test-case. Within a test-case, multiple tests can be run.
  void TestStart(const std::string &test_name, const std::string &test_configuration);
  void TestEnd();

  // Tests either an error count (should be zero) or two error codes (must match)
  void TestErrorCount(const size_t errors, const size_t size, const Arguments<U> &args);
  void TestErrorCodes(const StatusCode clblas_status, const StatusCode clblast_status,
                      const Arguments<U> &args);

  // Returns the number of failed tests
  size_t NumFailedTests() const { return tests_failed_; }

 protected:

  // The help-message
  std::string help_;

  // The OpenCL objects (accessible by derived classes)
  Platform platform_;
  Device device_;
  Context context_;
  Queue queue_;

  // Whether or not to run the full test-suite or just a smoke test
  const bool full_test_;

  // Whether or not to print extra information when testing
  const bool verbose_;

  // Retrieves the offset values to test with
  const std::vector<size_t> GetOffsets() const;

  // Retrieves the list of options as a string
  std::string GetOptionsString(const Arguments<U> &args); // for regular tests
  std::string GetSizesString(const Arguments<U> &args); // for invalid buffer sizes

  // Testing against reference implementations
  int compare_cblas_;
  int compare_clblas_;
  int compare_cublas_;

 private:

  // Internal methods to report a passed, skipped, or failed test
  void ReportPass();
  void ReportSkipped();
  void ReportError(const ErrorLogEntry &log_entry);

  // Prints the error or success symbol to screen
  void PrintTestResult(const std::string &message);

  // Prints an error log
  void PrintErrorLog(const std::vector<ErrorLogEntry> &error_log);

  // Logging and counting occurrences of errors
  std::vector<ErrorLogEntry> error_log_;
  size_t num_passed_;
  size_t num_skipped_;
  size_t num_failed_;

  // Counting the amount of errors printed on this row
  size_t print_count_;

  // Counting the number of test-cases with and without failures
  size_t tests_passed_;
  size_t tests_skipped_;
  size_t tests_failed_;

  // Arguments relevant for a specific routine
  std::vector<std::string> options_;
};

// Maximum number of test results printed on a single line
template <typename T, typename U> const size_t Tester<T,U>::kResultsPerLine = size_t{64};

// Error percentage is not applicable: error was caused by an incorrect status
template <typename T, typename U> const float Tester<T,U>::kStatusError = -1.0f;

// Constants holding start and end strings for terminal-output in colour
#if defined(_WIN32)
  template <typename T, typename U> const std::string Tester<T,U>::kPrintError = "";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintSuccess = "";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintWarning = "";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintMessage = "";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintEnd = "";
#else
  template <typename T, typename U> const std::string Tester<T,U>::kPrintError = "\x1b[31m";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintSuccess = "\x1b[32m";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintWarning = "\x1b[35m";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintMessage = "\x1b[1m";
  template <typename T, typename U> const std::string Tester<T,U>::kPrintEnd = "\x1b[0m";
#endif

// Sets the output error coding
#if defined(_WIN32)
  template <typename T, typename U> const std::string Tester<T,U>::kSuccessData = ":"; // success
  template <typename T, typename U> const std::string Tester<T,U>::kSuccessStatus = "."; // success
  template <typename T, typename U> const std::string Tester<T,U>::kErrorData = "X"; // error
  template <typename T, typename U> const std::string Tester<T,U>::kErrorStatus = "/"; // error
  template <typename T, typename U> const std::string Tester<T,U>::kSkippedCompilation = "\\"; // warning
  template <typename T, typename U> const std::string Tester<T,U>::kUnsupportedPrecision = "o"; // warning
  template <typename T, typename U> const std::string Tester<T,U>::kUnsupportedReference = "-"; // warning
#else
  template <typename T, typename U> const std::string Tester<T,U>::kSuccessData = "\x1b[32m:\x1b[0m"; // success
  template <typename T, typename U> const std::string Tester<T,U>::kSuccessStatus = "\x1b[32m.\x1b[0m"; // success
  template <typename T, typename U> const std::string Tester<T,U>::kErrorData = "\x1b[31mX\x1b[0m"; // error
  template <typename T, typename U> const std::string Tester<T,U>::kErrorStatus = "\x1b[31m/\x1b[0m"; // error
  template <typename T, typename U> const std::string Tester<T,U>::kSkippedCompilation = "\x1b[35m\\\x1b[0m"; // warning
  template <typename T, typename U> const std::string Tester<T,U>::kUnsupportedPrecision = "\x1b[35mo\x1b[0m"; // warning
  template <typename T, typename U> const std::string Tester<T,U>::kUnsupportedReference = "\x1b[35m-\x1b[0m"; // warning
#endif

// =================================================================================================
// Below are the non-member functions (separated because of otherwise required partial class
// template specialization)
// =================================================================================================

// Error margins
template <typename T> float getRelativeErrorMargin();
template <typename T> float getAbsoluteErrorMargin();
template <typename T> double getL2ErrorMargin();

// Compares two floating point values and returns whether they are within an acceptable error
// margin. This replaces GTest's EXPECT_NEAR().
template <typename T>
bool TestSimilarity(const T val1, const T val2);

// Retrieves a list of example scalar values, used for the alpha and beta arguments for the various
// routines. This function is specialised for the different data-types.
template <typename T>
const std::vector<T> GetExampleScalars(const bool full_test);

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_CORRECTNESS_TESTER_H_
#endif
