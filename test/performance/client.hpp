
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
// Typename T: the data-type of the routine's memory buffers (==precision)
// Typename U: the data-type of the alpha and beta arguments
//
// This file also provides the common interface to the performance client (see the 'RunClient'
// function for details).
//
// =================================================================================================

#ifndef CLBLAST_TEST_PERFORMANCE_CLIENT_H_
#define CLBLAST_TEST_PERFORMANCE_CLIENT_H_

#include <string>
#include <vector>
#include <utility>

#include "test/test_utilities.hpp"

// The libraries to test
#ifdef CLBLAST_REF_CLBLAS
  #include <clBLAS.h>
#endif
#include "test/wrapper_cuda.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Client {
 public:
  static const int kSeed;
  struct TimeResult { double minimum; double maximum; double mean; double standard_deviation; };

  // Shorthand for the routine-specific functions passed to the tester
  using Routine = std::function<StatusCode(const Arguments<U>&, Buffers<T>&, Queue&)>;
  using Reference1 = std::function<StatusCode(const Arguments<U>&, Buffers<T>&, Queue&)>;
  using Reference2 = std::function<StatusCode(const Arguments<U>&, BuffersHost<T>&, Queue&)>;
  using Reference3 = std::function<StatusCode(const Arguments<U>&, BuffersCUDA<T>&, Queue&)>;
  using SetMetric = std::function<void(Arguments<U>&, Queue&)>;
  using GetMetric = std::function<size_t(const Arguments<U>&)>;

  // The constructor
  Client(const Routine run_routine, const Reference1 run_reference1, const Reference2 run_reference2,
         const Reference3 run_reference3, const std::vector<std::string> &options,
         const std::vector<std::string> &buffers_in, const std::vector<std::string> &buffers_out,
         const GetMetric get_flops, const GetMetric get_bytes);

  // Parses all command-line arguments, filling in the arguments structure. If no command-line
  // argument is given for a particular argument, it is filled in with a default value.
  Arguments<U> ParseArguments(int argc, char *argv[], const size_t level,
                              const GetMetric default_a_ld,
                              const GetMetric default_b_ld,
                              const GetMetric default_c_ld);

  // The main client function, setting-up arguments, matrices, OpenCL buffers, etc. After set-up, it
  // calls the client routines.
  void PerformanceTest(Arguments<U> &args, const SetMetric set_sizes);

 private:

  // Runs a function a given number of times and returns the execution time of the shortest instance
  template <typename BufferType, typename RoutineType>
  TimeResult TimedExecution(const size_t num_runs, const Arguments<U> &args, BufferType &buffers,
                            Queue &queue, RoutineType run_blas, const std::string &library_name);

  // Prints the header of a performance-data table
  void PrintTableHeader(const Arguments<U>& args);

  // Prints a row of performance data, including results of two libraries
  void PrintTableRow(const Arguments<U>& args,
                     const std::vector<std::pair<std::string, TimeResult>>& timings);

  // The routine-specific functions passed to the tester
  const Routine run_routine_;
  const Reference1 run_reference1_;
  const Reference2 run_reference2_;
  const Reference3 run_reference3_;
  const std::vector<std::string> options_;
  const std::vector<std::string> buffers_in_;
  const std::vector<std::string> buffers_out_;
  const GetMetric get_flops_;
  const GetMetric get_bytes_;

  // Extra arguments
  bool warm_up_; // if enabled, do a warm-up run first before measuring execution time
};

// =================================================================================================

// Bogus reference function, in case a comparison library is not available
template <typename T, typename U, typename BufferType>
static StatusCode ReferenceNotAvailable(const Arguments<U> &, BufferType &, Queue &) {
  return StatusCode::kNotImplemented;
}

// The interface to the performance client. This is a separate function in the header such that it
// is automatically compiled for each routine, templated by the parameter "C".
template <typename C, typename T, typename U>
void RunClient(int argc, char *argv[]) {

  // Sets the reference to test against
  #ifdef CLBLAST_REF_CLBLAS
    auto reference1 = C::RunReference1; // clBLAS when available
  #else
    auto reference1 = ReferenceNotAvailable<T,U,Buffers<T>>;
  #endif
  #ifdef CLBLAST_REF_CBLAS
    auto reference2 = C::RunReference2; // CBLAS when available
  #else
    auto reference2 = ReferenceNotAvailable<T,U,BuffersHost<T>>;
  #endif
  #ifdef CLBLAST_REF_CUBLAS
    auto reference3 = C::RunReference3; // cuBLAS when available
  #else
    auto reference3 = ReferenceNotAvailable<T,U,BuffersCUDA<T>>;
  #endif

  // Creates a new client
  auto client = Client<T,U>(C::RunRoutine, reference1, reference2, reference3, C::GetOptions(),
                            C::BuffersIn(), C::BuffersOut(), C::GetFlops, C::GetBytes);

  // Simple command line argument parser with defaults
  auto args = client.ParseArguments(argc, argv, C::BLASLevel(),
                                    C::DefaultLDA, C::DefaultLDB, C::DefaultLDC);
  if (args.print_help) { return; }

  // Runs the client
  client.PerformanceTest(args, C::SetSizes);
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_PERFORMANCE_CLIENT_H_
#endif
