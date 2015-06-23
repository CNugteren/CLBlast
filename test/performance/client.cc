
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common functions for the client-test environment.
//
// =================================================================================================

#include "performance/client.h"

#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

namespace clblast {
// =================================================================================================

// This is the vector-vector variant of the set-up/tear-down client routine.
template <typename T>
void ClientXY(int argc, char *argv[], Routine2<T> client_routine,
              const std::vector<std::string> &options) {

  // Function to determine how to find the default value of the leading dimension of matrix A.
  // Note: this is not relevant for this client but given anyway.
  auto default_ld_a = [](const Arguments<T> args) { return args.n; };

  // Simple command line argument parser with defaults
  auto args = ParseArguments<T>(argc, argv, options, default_ld_a);
  if (args.print_help) { return; }

  // Prints the header of the output table
  PrintTableHeader(args.silent, options);

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, kDeviceType, args.device_id);
  auto context = Context(device);
  auto queue = CommandQueue(context, device);
  if (args.compare_clblas) { clblasSetup(); }

  // Iterates over all "num_step" values jumping by "step" each time
  auto s = size_t{0};
  while(true) {

    // Computes the data sizes
    auto x_size = args.n*args.x_inc + args.x_offset;
    auto y_size = args.n*args.y_inc + args.y_offset;

    // Populates input host vectors with random data
    std::vector<T> x_source(x_size);
    std::vector<T> y_source(y_size);
    PopulateVector(x_source);
    PopulateVector(y_source);

    // Creates the vectors on the device
    auto x_buffer = Buffer(context, CL_MEM_READ_WRITE, x_size*sizeof(T));
    auto y_buffer = Buffer(context, CL_MEM_READ_WRITE, y_size*sizeof(T));
    x_buffer.WriteBuffer(queue, x_size*sizeof(T), x_source);
    y_buffer.WriteBuffer(queue, y_size*sizeof(T), y_source);

    // Runs the routine-specific code
    client_routine(args, x_buffer, y_buffer, queue);

    // Makes the jump to the next step
    ++s;
    if (s >= args.num_steps) { break; }
    args.n += args.step;
  }

  // Cleans-up and returns
  if (args.compare_clblas) { clblasTeardown(); }
}

// Compiles the above function
template void ClientXY<float>(int, char **, Routine2<float>, const std::vector<std::string>&);
template void ClientXY<double>(int, char **, Routine2<double>, const std::vector<std::string>&);
template void ClientXY<float2>(int, char **, Routine2<float2>, const std::vector<std::string>&);
template void ClientXY<double2>(int, char **, Routine2<double2>, const std::vector<std::string>&);

// =================================================================================================

// This is the matrix-vector-vector variant of the set-up/tear-down client routine.
template <typename T>
void ClientAXY(int argc, char *argv[], Routine3<T> client_routine,
               const std::vector<std::string> &options) {

  // Function to determine how to find the default value of the leading dimension of matrix A
  auto default_ld_a = [](const Arguments<T> args) { return args.n; };

  // Simple command line argument parser with defaults
  auto args = ParseArguments<T>(argc, argv, options, default_ld_a);
  if (args.print_help) { return; }

  // Prints the header of the output table
  PrintTableHeader(args.silent, options);

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, kDeviceType, args.device_id);
  auto context = Context(device);
  auto queue = CommandQueue(context, device);
  if (args.compare_clblas) { clblasSetup(); }

  // Iterates over all "num_step" values jumping by "step" each time
  auto s = size_t{0};
  while(true) {

    // Computes the second dimension of the matrix taking the rotation into account
    auto a_two = (args.layout == Layout::kRowMajor) ? args.m : args.n;

    // Computes the vector sizes in case the matrix is transposed
    auto a_transposed = (args.a_transpose != Transpose::kNo);
    auto m_real = (a_transposed) ? args.n : args.m;
    auto n_real = (a_transposed) ? args.m : args.n;

    // Computes the data sizes
    auto a_size = a_two * args.a_ld + args.a_offset;
    auto x_size = n_real*args.x_inc + args.x_offset;
    auto y_size = m_real*args.y_inc + args.y_offset;

    // Populates input host vectors with random data
    std::vector<T> a_source(a_size);
    std::vector<T> x_source(x_size);
    std::vector<T> y_source(y_size);
    PopulateVector(a_source);
    PopulateVector(x_source);
    PopulateVector(y_source);

    // Creates the vectors on the device
    auto a_buffer = Buffer(context, CL_MEM_READ_WRITE, a_size*sizeof(T));
    auto x_buffer = Buffer(context, CL_MEM_READ_WRITE, x_size*sizeof(T));
    auto y_buffer = Buffer(context, CL_MEM_READ_WRITE, y_size*sizeof(T));
    a_buffer.WriteBuffer(queue, a_size*sizeof(T), a_source);
    x_buffer.WriteBuffer(queue, x_size*sizeof(T), x_source);
    y_buffer.WriteBuffer(queue, y_size*sizeof(T), y_source);

    // Runs the routine-specific code
    client_routine(args, a_buffer, x_buffer, y_buffer, queue);

    // Makes the jump to the next step
    ++s;
    if (s >= args.num_steps) { break; }
    args.m += args.step;
    args.n += args.step;
    args.a_ld += args.step;
  }

  // Cleans-up and returns
  if (args.compare_clblas) { clblasTeardown(); }
}

// Compiles the above function
template void ClientAXY<float>(int, char **, Routine3<float>, const std::vector<std::string>&);
template void ClientAXY<double>(int, char **, Routine3<double>, const std::vector<std::string>&);
template void ClientAXY<float2>(int, char **, Routine3<float2>, const std::vector<std::string>&);
template void ClientAXY<double2>(int, char **, Routine3<double2>, const std::vector<std::string>&);

// =================================================================================================

// This is the matrix-matrix variant of the set-up/tear-down client routine.
template <typename T>
void ClientAC(int argc, char *argv[], Routine2<T> client_routine,
              const std::vector<std::string> &options) {

  // Function to determine how to find the default value of the leading dimension of matrix A
  auto default_ld_a = [](const Arguments<T> args) { return args.k; };

  // Simple command line argument parser with defaults
  auto args = ParseArguments<T>(argc, argv, options, default_ld_a);
  if (args.print_help) { return; }

  // Prints the header of the output table
  PrintTableHeader(args.silent, options);

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, kDeviceType, args.device_id);
  auto context = Context(device);
  auto queue = CommandQueue(context, device);
  if (args.compare_clblas) { clblasSetup(); }

  // Computes whether or not the matrices are transposed. Note that we assume a default of
  // column-major and no-transpose. If one of them is different (but not both), then rotated
  // is considered true.
  auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                   (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);

  // Iterates over all "num_step" values jumping by "step" each time
  auto s = size_t{0};
  while(true) {

    // Computes the data sizes
    auto a_two = (a_rotated) ? args.n : args.k;
    auto a_size = a_two * args.a_ld + args.a_offset;
    auto c_size = args.n * args.c_ld + args.c_offset;

    // Populates input host matrices with random data
    std::vector<T> a_source(a_size);
    std::vector<T> c_source(c_size);
    PopulateVector(a_source);
    PopulateVector(c_source);

    // Creates the matrices on the device
    auto a_buffer = Buffer(context, CL_MEM_READ_WRITE, a_size*sizeof(T));
    auto c_buffer = Buffer(context, CL_MEM_READ_WRITE, c_size*sizeof(T));
    a_buffer.WriteBuffer(queue, a_size*sizeof(T), a_source);
    c_buffer.WriteBuffer(queue, c_size*sizeof(T), c_source);

    // Runs the routine-specific code
    client_routine(args, a_buffer, c_buffer, queue);

    // Makes the jump to the next step
    ++s;
    if (s >= args.num_steps) { break; }
    args.n += args.step;
    args.k += args.step;
    args.a_ld += args.step;
    args.c_ld += args.step;
  }

  // Cleans-up and returns
  if (args.compare_clblas) { clblasTeardown(); }
}

// Compiles the above function
template void ClientAC<float>(int, char **, Routine2<float>, const std::vector<std::string>&);
template void ClientAC<double>(int, char **, Routine2<double>, const std::vector<std::string>&);
template void ClientAC<float2>(int, char **, Routine2<float2>, const std::vector<std::string>&);
template void ClientAC<double2>(int, char **, Routine2<double2>, const std::vector<std::string>&);

// =================================================================================================

// This is the matrix-matrix-matrix variant of the set-up/tear-down client routine.
template <typename T>
void ClientABC(int argc, char *argv[], Routine3<T> client_routine,
               const std::vector<std::string> &options) {

  // Function to determine how to find the default value of the leading dimension of matrix A
  auto default_ld_a = [](const Arguments<T> args) { return args.m; };

  // Simple command line argument parser with defaults
  auto args = ParseArguments<T>(argc, argv, options, default_ld_a);
  if (args.print_help) { return; }

  // Prints the header of the output table
  PrintTableHeader(args.silent, options);

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, kDeviceType, args.device_id);
  auto context = Context(device);
  auto queue = CommandQueue(context, device);
  if (args.compare_clblas) { clblasSetup(); }

  // Computes whether or not the matrices are transposed. Note that we assume a default of
  // column-major and no-transpose. If one of them is different (but not both), then rotated
  // is considered true.
  auto a_rotated = (args.layout == Layout::kColMajor && args.a_transpose != Transpose::kNo) ||
                   (args.layout == Layout::kRowMajor && args.a_transpose == Transpose::kNo);
  auto b_rotated = (args.layout == Layout::kColMajor && args.b_transpose != Transpose::kNo) ||
                   (args.layout == Layout::kRowMajor && args.b_transpose == Transpose::kNo);
  auto c_rotated = (args.layout == Layout::kRowMajor);

  // Iterates over all "num_step" values jumping by "step" each time
  auto s = size_t{0};
  while(true) {

    // Computes the data sizes
    auto a_two = (a_rotated) ? args.m : args.k;
    auto b_two = (b_rotated) ? args.k : args.n;
    auto c_two = (c_rotated) ? args.m : args.n;
    auto a_size = a_two * args.a_ld + args.a_offset;
    auto b_size = b_two * args.b_ld + args.b_offset;
    auto c_size = c_two * args.c_ld + args.c_offset;

    // Populates input host matrices with random data
    std::vector<T> a_source(a_size);
    std::vector<T> b_source(b_size);
    std::vector<T> c_source(c_size);
    PopulateVector(a_source);
    PopulateVector(b_source);
    PopulateVector(c_source);

    // Creates the matrices on the device
    auto a_buffer = Buffer(context, CL_MEM_READ_WRITE, a_size*sizeof(T));
    auto b_buffer = Buffer(context, CL_MEM_READ_WRITE, b_size*sizeof(T));
    auto c_buffer = Buffer(context, CL_MEM_READ_WRITE, c_size*sizeof(T));
    a_buffer.WriteBuffer(queue, a_size*sizeof(T), a_source);
    b_buffer.WriteBuffer(queue, b_size*sizeof(T), b_source);
    c_buffer.WriteBuffer(queue, c_size*sizeof(T), c_source);

    // Runs the routine-specific code
    client_routine(args, a_buffer, b_buffer, c_buffer, queue);

    // Makes the jump to the next step
    ++s;
    if (s >= args.num_steps) { break; }
    args.m += args.step;
    args.n += args.step;
    args.k += args.step;
    args.a_ld += args.step;
    args.b_ld += args.step;
    args.c_ld += args.step;
  }

  // Cleans-up and returns
  if (args.compare_clblas) { clblasTeardown(); }
}

// Compiles the above function
template void ClientABC<float>(int, char **, Routine3<float>, const std::vector<std::string>&);
template void ClientABC<double>(int, char **, Routine3<double>, const std::vector<std::string>&);
template void ClientABC<float2>(int, char **, Routine3<float2>, const std::vector<std::string>&);
template void ClientABC<double2>(int, char **, Routine3<double2>, const std::vector<std::string>&);

// =================================================================================================

// Parses all arguments available for the CLBlast client testers. Some arguments might not be
// applicable, but are searched for anyway to be able to create one common argument parser. All
// arguments have a default value in case they are not found.
template <typename T>
Arguments<T> ParseArguments(int argc, char *argv[], const std::vector<std::string> &options,
                            const std::function<size_t(const Arguments<T>)> default_ld_a) {
  auto args = Arguments<T>{};
  auto help = std::string{"Options given/available:\n"};

  // These are the options which are not for every client: they are optional
  for (auto &o: options) {

    // Data-sizes
    if (o == kArgM) { args.m = args.k  = GetArgument(argc, argv, help, kArgM, 512UL); }
    if (o == kArgN) { args.n           = GetArgument(argc, argv, help, kArgN, 512UL); }
    if (o == kArgK) { args.k           = GetArgument(argc, argv, help, kArgK, 512UL); }

    // Data-layouts
    if (o == kArgLayout)   { args.layout      = GetArgument(argc, argv, help, kArgLayout, Layout::kRowMajor); }
    if (o == kArgATransp)  { args.a_transpose = GetArgument(argc, argv, help, kArgATransp, Transpose::kNo); }
    if (o == kArgBTransp)  { args.b_transpose = GetArgument(argc, argv, help, kArgBTransp, Transpose::kNo); }
    if (o == kArgSide)     { args.side        = GetArgument(argc, argv, help, kArgSide, Side::kLeft); }
    if (o == kArgTriangle) { args.triangle    = GetArgument(argc, argv, help, kArgTriangle, Triangle::kUpper); }

    // Vector arguments
    if (o == kArgXInc)    { args.x_inc    = GetArgument(argc, argv, help, kArgXInc, size_t{1}); }
    if (o == kArgYInc)    { args.y_inc    = GetArgument(argc, argv, help, kArgYInc, size_t{1}); }
    if (o == kArgXOffset) { args.x_offset = GetArgument(argc, argv, help, kArgXOffset, size_t{0}); }
    if (o == kArgYOffset) { args.y_offset = GetArgument(argc, argv, help, kArgYOffset, size_t{0}); }

    // Matrix arguments
    if (o == kArgALeadDim) { args.a_ld     = GetArgument(argc, argv, help, kArgALeadDim, default_ld_a(args)); }
    if (o == kArgBLeadDim) { args.b_ld     = GetArgument(argc, argv, help, kArgBLeadDim, args.n); }
    if (o == kArgCLeadDim) { args.c_ld     = GetArgument(argc, argv, help, kArgCLeadDim, args.n); }
    if (o == kArgAOffset)  { args.a_offset = GetArgument(argc, argv, help, kArgAOffset, size_t{0}); }
    if (o == kArgBOffset)  { args.b_offset = GetArgument(argc, argv, help, kArgBOffset, size_t{0}); }
    if (o == kArgCOffset)  { args.c_offset = GetArgument(argc, argv, help, kArgCOffset, size_t{0}); }

    // Scalar values 
    if (o == kArgAlpha) { args.alpha = GetArgument(argc, argv, help, kArgAlpha, GetScalar<T>()); }
    if (o == kArgBeta)  { args.beta  = GetArgument(argc, argv, help, kArgBeta, GetScalar<T>()); }
  }

  // These are the options common to all routines
  args.platform_id    = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  args.device_id      = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  args.precision      = GetArgument(argc, argv, help, kArgPrecision, Precision::kSingle);
  args.compare_clblas = GetArgument(argc, argv, help, kArgCompareclblas, true);
  args.step           = GetArgument(argc, argv, help, kArgStepSize, size_t{1});
  args.num_steps      = GetArgument(argc, argv, help, kArgNumSteps, size_t{0});
  args.num_runs       = GetArgument(argc, argv, help, kArgNumRuns, size_t{10});
  args.print_help     = CheckArgument(argc, argv, help, kArgHelp);
  args.silent         = CheckArgument(argc, argv, help, kArgQuiet);
  args.no_abbrv       = CheckArgument(argc, argv, help, kArgNoAbbreviations);

  // Prints the chosen (or defaulted) arguments to screen. This also serves as the help message,
  // which is thus always displayed (unless silence is specified).
  if (!args.silent) { fprintf(stdout, "%s\n", help.c_str()); }

  // Returns the arguments
  return args;
}

// =================================================================================================

// Creates a vector of timing results, filled with execution times of the 'main computation'. The
// timing is performed using the milliseconds chrono functions. The function returns the minimum
// value found in the vector of timing results. The return value is in milliseconds.
double TimedExecution(const size_t num_runs, std::function<void()> main_computation) {
  auto timings = std::vector<double>(num_runs);
  for (auto &timing: timings) {
    auto start_time = std::chrono::steady_clock::now();

    // Executes the main computation
    main_computation();

    // Records and stores the end-time
    auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
  }
  return *std::min_element(timings.begin(), timings.end());
}

// =================================================================================================

// Prints the header of the performance table
void PrintTableHeader(const bool silent, const std::vector<std::string> &args) {
  if (!silent) {
    for (auto i=size_t{0}; i<args.size(); ++i) { fprintf(stdout, "%9s ", ""); }
    fprintf(stdout, " | <--       CLBlast       --> | <--      clBLAS      --> |\n");
  }
  for (auto &argument: args) { fprintf(stdout, "%9s;", argument.c_str()); }
  fprintf(stdout, "%9s;%9s;%9s;%9s;%9s;%9s\n",
          "ms_1", "GFLOPS_1", "GBs_1", "ms_2", "GFLOPS_2", "GBs_2");
}

// Print a performance-result row
void PrintTableRow(const std::vector<size_t> &args_int, const std::vector<std::string> &args_string,
                   const bool no_abbrv, const double ms_clblast, const double ms_clblas,
                   const unsigned long long flops, const unsigned long long bytes) {

  // Computes the GFLOPS and GB/s metrics
  auto gflops_clblast = (ms_clblast != 0.0) ? (flops*1e-6)/ms_clblast : 0;
  auto gflops_clblas = (ms_clblas != 0.0) ? (flops*1e-6)/ms_clblas: 0;
  auto gbs_clblast = (ms_clblast != 0.0) ? (bytes*1e-6)/ms_clblast : 0;
  auto gbs_clblas = (ms_clblas != 0.0) ? (bytes*1e-6)/ms_clblas: 0;

  // Outputs the argument values
  for (auto &argument: args_int) {
    if (!no_abbrv && argument >= 1024*1024 && IsMultiple(argument, 1024*1024)) {
      fprintf(stdout, "%8luM;", argument/(1024*1024));
    }
    else if (!no_abbrv && argument >= 1024 && IsMultiple(argument, 1024)) {
      fprintf(stdout, "%8luK;", argument/1024);
    }
    else {
      fprintf(stdout, "%9lu;", argument);
    }
  }
  for (auto &argument: args_string) {
    fprintf(stdout, "%9s;", argument.c_str());
  }

  // Outputs the performance numbers
  fprintf(stdout, "%9.2lf;%9.1lf;%9.1lf;%9.2lf;%9.1lf;%9.1lf\n",
          ms_clblast, gflops_clblast, gbs_clblast,
          ms_clblas, gflops_clblas, gbs_clblas);
}

// =================================================================================================
} // namespace clblast
