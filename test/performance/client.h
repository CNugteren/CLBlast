
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This class implements the performance-test client. It is generic for all CLBlast routines by
// taking a number of routine-specific functions as arguments, such as how to compute buffer sizes
// or how to get the FLOPS count.
//
// This file also provides the common interface to the performance client (see the 'RunClient'
// function for details).
//
// =================================================================================================

#ifndef CLBLAST_TEST_PERFORMANCE_CLIENT_H_
#define CLBLAST_TEST_PERFORMANCE_CLIENT_H_

#include <string>
#include <vector>

// The libraries to test
#include <clBLAS.h>
#include "clblast.h"

#include "internal/utilities.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Client {
 public:

  // Types of devices to consider
  const cl_device_type kDeviceType = CL_DEVICE_TYPE_ALL;

  // Shorthand for the routine-specific functions passed to the tester
  using Routine = std::function<StatusCode(const Arguments<T>&, const Buffers&, CommandQueue&)>;
  using SetMetric = std::function<void(Arguments<T>&)>;
  using GetMetric = std::function<size_t(const Arguments<T>&)>;

  // The constructor
  Client(const Routine run_routine, const Routine run_reference,
         const std::vector<std::string> &options,
         const GetMetric get_flops, const GetMetric get_bytes);

  // Parses all command-line arguments, filling in the arguments structure. If no command-line
  // argument is given for a particular argument, it is filled in with a default value.
  Arguments<T> ParseArguments(int argc, char *argv[], const GetMetric default_a_ld,
                              const GetMetric default_b_ld, const GetMetric default_c_ld);

  // The main client function, setting-up arguments, matrices, OpenCL buffers, etc. After set-up, it
  // calls the client routines.
  void PerformanceTest(Arguments<T> &args, const SetMetric set_sizes);

 private:

  // Runs a function a given number of times and returns the execution time of the shortest instance
  double TimedExecution(const size_t num_runs, const Arguments<T> &args, const Buffers &buffers,
                        CommandQueue &queue, Routine run_blas, const std::string &library_name);

  // Prints the header of a performance-data table
  void PrintTableHeader(const bool silent, const std::vector<std::string> &args);

  // Prints a row of performance data, including results of two libraries
  void PrintTableRow(const Arguments<T>& args, const double ms_clblast, const double ms_clblas);

  // The routine-specific functions passed to the tester
  const Routine run_routine_;
  const Routine run_reference_;
  const std::vector<std::string> options_;
  const GetMetric get_flops_;
  const GetMetric get_bytes_;
};

// =================================================================================================

// The interface to the performance client. This is a separate function in the header such that it
// is automatically compiled for each routine, templated by the parameter "C".
template <typename C, typename T>
void RunClient(int argc, char *argv[]) {

  // Creates a new client
  auto client = Client<T>(C::RunRoutine, C::RunReference, C::GetOptions(),
                          C::GetFlops, C::GetBytes);

  // Simple command line argument parser with defaults
  auto args = client.ParseArguments(argc, argv, C::DefaultLDA, C::DefaultLDB, C::DefaultLDC);
  if (args.print_help) { return; }

  // Runs the client
  client.PerformanceTest(args, C::SetSizes);
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_PERFORMANCE_CLIENT_H_
#endif
