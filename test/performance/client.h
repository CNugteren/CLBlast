
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides common function declarations to be used with the test clients.
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

// Types of devices to consider
const cl_device_type kDeviceType = CL_DEVICE_TYPE_ALL;

// =================================================================================================

// Shorthand for a BLAS routine with 2 or 3 OpenCL buffers as argument
template <typename T>
using Routine2 = std::function<void(const Arguments<T>&,
                                    const Buffer&, const Buffer&,
                                    CommandQueue&)>;
template <typename T>
using Routine3 = std::function<void(const Arguments<T>&,
                                    const Buffer&, const Buffer&, const Buffer&,
                                    CommandQueue&)>;

// =================================================================================================

// These are the main client functions, setting-up arguments, matrices, OpenCL buffers, etc. After
// set-up, they call the client routine, passed as argument to this function.
template <typename T>
void ClientXY(int argc, char *argv[], Routine2<T> client_routine,
              const std::vector<std::string> &options);
template <typename T>
void ClientAXY(int argc, char *argv[], Routine3<T> client_routine,
               const std::vector<std::string> &options);
template <typename T>
void ClientAC(int argc, char *argv[], Routine2<T> client_routine,
              const std::vector<std::string> &options);
template <typename T>
void ClientABC(int argc, char *argv[], Routine3<T> client_routine,
               const std::vector<std::string> &options, const bool symmetric);

// =================================================================================================

// Parses all command-line arguments, filling in the arguments structure. If no command-line
// argument is given for a particular argument, it is filled in with a default value.
template <typename T>
Arguments<T> ParseArguments(int argc, char *argv[], const std::vector<std::string> &options,
                            const std::function<size_t(const Arguments<T>)> default_ld_a);

// Retrieves only the precision command-line argument, since the above function is templated based
// on the precision
Precision GetPrecision(int argc, char *argv[]);

// =================================================================================================

// Runs a function a given number of times and returns the execution time of the shortest instance
double TimedExecution(const size_t num_runs, std::function<void()> main_computation);

// =================================================================================================

// Prints the header of a performance-data table
void PrintTableHeader(const bool silent, const std::vector<std::string> &args);

// Prints a row of performance data, including results of two libraries
void PrintTableRow(const std::vector<size_t> &args_int, const std::vector<std::string> &args_string,
                   const bool abbreviations, const double ms_clblast, const double ms_clblas,
                   const unsigned long long flops, const unsigned long long bytes);

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_PERFORMANCE_CLIENT_H_
#endif
