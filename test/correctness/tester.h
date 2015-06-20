
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
//
// =================================================================================================

#ifndef CLBLAST_TEST_CORRECTNESS_TESTER_H_
#define CLBLAST_TEST_CORRECTNESS_TESTER_H_

#include <string>
#include <vector>
#include <memory>

// The libraries
#include <clBLAS.h>
#include "clblast.h"

#include "internal/utilities.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Tester {
 public:

  // Types of devices to consider
  const cl_device_type kDeviceType = CL_DEVICE_TYPE_ALL;

  // Maximum number of test results printed on a single line
  static constexpr auto kResultsPerLine = size_t{64};

  // Error percentage is not applicable: error was caused by an incorrect status
  static constexpr auto kStatusError = -1.0f;

  // Set the allowed error margin for floating-point comparisons
  static constexpr auto kErrorMarginRelative = 1.0e-2;
  static constexpr auto kErrorMarginAbsolute = 1.0e-10;

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

  // The layouts and transpose-options to test with
  static const std::vector<Layout> kLayouts;
  static const std::vector<Transpose> kTransposes;

  // This structure combines the above log-entry with a status code an error percentage
  struct ErrorLogEntry {
    StatusCode status_expect;
    StatusCode status_found;
    float error_percentage;
    Arguments<T> args;
  };

  // Creates an instance of the tester, running on a particular OpenCL platform and device. It
  // takes the routine's names as an additional parameter.
  explicit Tester(int argc, char *argv[], const bool silent,
                  const std::string &name, const std::vector<std::string> &options);
  ~Tester();

  // These methods start and end a test-case. Within a test-case, multiple tests can be run.
  void TestStart(const std::string &test_name, const std::string &test_configuration);
  void TestEnd();

  // Compares two floating point values for similarity. Allows for a certain relative error margin.
  static bool TestSimilarity(const T val1, const T val2);

  // Tests either an error count (should be zero) or two error codes (must match)
  void TestErrorCount(const size_t errors, const size_t size, const Arguments<T> &args);
  void TestErrorCodes(const StatusCode clblas_status, const StatusCode clblast_status,
                      const Arguments<T> &args);

 protected:

  // Retrieves a list of example scalars of the right type
  const std::vector<T> GetExampleScalars();

  // Retrieves a list of offset values to test
  const std::vector<size_t> GetOffsets();

  // The help-message
  std::string help_;

  // The OpenCL objects (accessible by derived classes)
  Platform platform_;
  Device device_;
  Context context_;
  CommandQueue queue_;

 private:

  // Internal methods to report a passed, skipped, or failed test
  void ReportPass();
  void ReportSkipped();
  void ReportError(const ErrorLogEntry &log_entry);

  // Prints the error or success symbol to screen
  void PrintTestResult(const std::string &message);

  // Whether or not to run the full test-suite or just a smoke test
  bool full_test_;

  // Logging and counting occurrences of errors
  std::vector<ErrorLogEntry> error_log_;
  size_t num_passed_;
  size_t num_skipped_;
  size_t num_errors_;

  // Counting the amount of errors printed on this row
  size_t print_count_;

  // Counting the number of test-cases with and without failures
  size_t tests_failed_;
  size_t tests_passed_;

  // Arguments relevant for a specific routine
  std::vector<std::string> options_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_CORRECTNESS_TESTER_H_
#endif
