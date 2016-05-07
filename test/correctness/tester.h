
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

// The libraries
#ifdef CLBLAST_REF_CLBLAS
  #include <clBLAS.h>
#endif
#include "clblast.h"

#include "internal/utilities.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Tester {
 public:

  // Maximum number of test results printed on a single line
  static constexpr auto kResultsPerLine = size_t{64};

  // Error percentage is not applicable: error was caused by an incorrect status
  static constexpr auto kStatusError = -1.0f;

  // Constants holding start and end strings for terminal-output in colour
  const std::string kPrintError{"\x1b[31m"};
  const std::string kPrintSuccess{"\x1b[32m"};
  const std::string kPrintWarning{"\x1b[35m"};
  const std::string kPrintMessage{"\x1b[1m"};
  const std::string kPrintEnd{"\x1b[0m"};

  // Sets the output error coding
  const std::string kSuccessData{kPrintSuccess + ":" + kPrintEnd};
  const std::string kSuccessStatus{kPrintSuccess + "." + kPrintEnd};
  const std::string kErrorData{kPrintError + "X" + kPrintEnd};
  const std::string kErrorStatus{kPrintError + "/" + kPrintEnd};
  const std::string kSkippedCompilation{kPrintWarning + "\\" + kPrintEnd};
  const std::string kUnsupportedPrecision{kPrintWarning + "o" + kPrintEnd};
  const std::string kUnsupportedReference{kPrintWarning + "." + kPrintEnd};

  // This structure combines the above log-entry with a status code an error percentage
  struct ErrorLogEntry {
    StatusCode status_expect;
    StatusCode status_found;
    float error_percentage;
    Arguments<U> args;
  };

  // Creates an instance of the tester, running on a particular OpenCL platform and device. It
  // takes the routine's names as an additional parameter.
  explicit Tester(int argc, char *argv[], const bool silent,
                  const std::string &name, const std::vector<std::string> &options);
  ~Tester();

  // These methods start and end a test-case. Within a test-case, multiple tests can be run.
  void TestStart(const std::string &test_name, const std::string &test_configuration);
  void TestEnd();

  // Tests either an error count (should be zero) or two error codes (must match)
  void TestErrorCount(const size_t errors, const size_t size, const Arguments<U> &args);
  void TestErrorCodes(const StatusCode clblas_status, const StatusCode clblast_status,
                      const Arguments<U> &args);

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

  // Testing against reference implementations
  int compare_cblas_;
  int compare_clblas_;

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

// =================================================================================================
// Below are the non-member functions (separated because of otherwise required partial class
// template specialization)
// =================================================================================================

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
