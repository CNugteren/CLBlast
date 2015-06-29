
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

// Constructor
template <typename T>
Client<T>::Client(const Routine run_routine, const Routine run_reference,
                  const std::vector<std::string> &options,
                  const GetMetric get_flops, const GetMetric get_bytes):
  run_routine_(run_routine),
  run_reference_(run_reference),
  options_(options),
  get_flops_(get_flops),
  get_bytes_(get_bytes) {
}

// =================================================================================================

// Parses all arguments available for the CLBlast client testers. Some arguments might not be
// applicable, but are searched for anyway to be able to create one common argument parser. All
// arguments have a default value in case they are not found.
template <typename T>
Arguments<T> Client<T>::ParseArguments(int argc, char *argv[], const GetMetric default_a_ld,
                                       const GetMetric default_b_ld, const GetMetric default_c_ld) {
  auto args = Arguments<T>{};
  auto help = std::string{"Options given/available:\n"};

  // These are the options which are not for every client: they are optional
  for (auto &o: options_) {

    // Data-sizes
    if (o == kArgM) { args.m  = GetArgument(argc, argv, help, kArgM, 512UL); }
    if (o == kArgN) { args.n  = GetArgument(argc, argv, help, kArgN, 512UL); }
    if (o == kArgK) { args.k  = GetArgument(argc, argv, help, kArgK, 512UL); }

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
    if (o == kArgALeadDim) { args.a_ld     = GetArgument(argc, argv, help, kArgALeadDim, default_a_ld(args)); }
    if (o == kArgBLeadDim) { args.b_ld     = GetArgument(argc, argv, help, kArgBLeadDim, default_b_ld(args)); }
    if (o == kArgCLeadDim) { args.c_ld     = GetArgument(argc, argv, help, kArgCLeadDim, default_c_ld(args)); }
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

// This is main performance tester
template <typename T>
void Client<T>::PerformanceTest(Arguments<T> &args, const SetMetric set_sizes) {

  // Prints the header of the output table
  PrintTableHeader(args.silent, options_);

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, kDeviceType, args.device_id);
  auto context = Context(device);
  auto queue = CommandQueue(context, device);
  if (args.compare_clblas) { clblasSetup(); }

  // Iterates over all "num_step" values jumping by "step" each time
  auto s = size_t{0};
  while(true) {

    // Sets the buffer sizes (routine-specific)
    set_sizes(args);

    // Populates input host matrices with random data
    std::vector<T> x_source(args.x_size);
    std::vector<T> y_source(args.y_size);
    std::vector<T> a_source(args.a_size);
    std::vector<T> b_source(args.b_size);
    std::vector<T> c_source(args.c_size);
    PopulateVector(x_source);
    PopulateVector(y_source);
    PopulateVector(a_source);
    PopulateVector(b_source);
    PopulateVector(c_source);

    // Creates the matrices on the device
    auto x_vec = Buffer(context, CL_MEM_READ_WRITE, args.x_size*sizeof(T));
    auto y_vec = Buffer(context, CL_MEM_READ_WRITE, args.y_size*sizeof(T));
    auto a_mat = Buffer(context, CL_MEM_READ_WRITE, args.a_size*sizeof(T));
    auto b_mat = Buffer(context, CL_MEM_READ_WRITE, args.b_size*sizeof(T));
    auto c_mat = Buffer(context, CL_MEM_READ_WRITE, args.c_size*sizeof(T));
    x_vec.WriteBuffer(queue, args.x_size*sizeof(T), x_source);
    y_vec.WriteBuffer(queue, args.y_size*sizeof(T), y_source);
    a_mat.WriteBuffer(queue, args.a_size*sizeof(T), a_source);
    b_mat.WriteBuffer(queue, args.b_size*sizeof(T), b_source);
    c_mat.WriteBuffer(queue, args.c_size*sizeof(T), c_source);
    auto buffers = Buffers{x_vec, y_vec, a_mat, b_mat, c_mat};

    // Runs the routines and collects the timings
    auto ms_clblast = TimedExecution(args.num_runs, args, buffers, queue, run_routine_, "CLBlast");
    auto ms_clblas = TimedExecution(args.num_runs, args, buffers, queue, run_reference_, "clBLAS");

    // Prints the performance of both libraries
    PrintTableRow(args, ms_clblast, ms_clblas);

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

// =================================================================================================

// Creates a vector of timing results, filled with execution times of the 'main computation'. The
// timing is performed using the milliseconds chrono functions. The function returns the minimum
// value found in the vector of timing results. The return value is in milliseconds.
template <typename T>
double Client<T>::TimedExecution(const size_t num_runs, const Arguments<T> &args,
                                 const Buffers &buffers, CommandQueue &queue,
                                 Routine run_blas, const std::string &library_name) {
  auto timings = std::vector<double>(num_runs);
  for (auto &timing: timings) {
    auto start_time = std::chrono::steady_clock::now();

    // Executes the main computation
    auto status = run_blas(args, buffers, queue);
    if (status != StatusCode::kSuccess) {
      throw std::runtime_error(library_name+" error: "+ToString(static_cast<int>(status)));
    }

    // Records and stores the end-time
    auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
  }
  return *std::min_element(timings.begin(), timings.end());
}

// =================================================================================================

// Prints the header of the performance table
template <typename T>
void Client<T>::PrintTableHeader(const bool silent, const std::vector<std::string> &args) {
  if (!silent) {
    for (auto i=size_t{0}; i<args.size(); ++i) { fprintf(stdout, "%9s ", ""); }
    fprintf(stdout, " | <--       CLBlast       --> | <--      clBLAS      --> |\n");
  }
  for (auto &argument: args) { fprintf(stdout, "%9s;", argument.c_str()); }
  fprintf(stdout, "%9s;%9s;%9s;%9s;%9s;%9s\n",
          "ms_1", "GFLOPS_1", "GBs_1", "ms_2", "GFLOPS_2", "GBs_2");
}

// Print a performance-result row
template <typename T>
void Client<T>::PrintTableRow(const Arguments<T>& args, const double ms_clblast,
                              const double ms_clblas) {

  // Creates a vector of relevant variables
  auto integers = std::vector<size_t>{};
  for (auto &o: options_) {
    if      (o == kArgM) {        integers.push_back(args.m); }
    if      (o == kArgN) {        integers.push_back(args.n); }
    else if (o == kArgK) {        integers.push_back(args.k); }
    else if (o == kArgLayout) {   integers.push_back(static_cast<size_t>(args.layout)); }
    else if (o == kArgSide) {     integers.push_back(static_cast<size_t>(args.side)); }
    else if (o == kArgTriangle) { integers.push_back(static_cast<size_t>(args.triangle)); }
    else if (o == kArgATransp) {  integers.push_back(static_cast<size_t>(args.a_transpose)); }
    else if (o == kArgBTransp) {  integers.push_back(static_cast<size_t>(args.b_transpose)); }
    else if (o == kArgXInc) {     integers.push_back(args.x_inc); }
    else if (o == kArgYInc) {     integers.push_back(args.y_inc); }
    else if (o == kArgXOffset) {  integers.push_back(args.x_offset); }
    else if (o == kArgYOffset) {  integers.push_back(args.y_offset); }
    else if (o == kArgALeadDim) { integers.push_back(args.a_ld); }
    else if (o == kArgBLeadDim) { integers.push_back(args.b_ld); }
    else if (o == kArgCLeadDim) { integers.push_back(args.c_ld); }
    else if (o == kArgAOffset) {  integers.push_back(args.a_offset); }
    else if (o == kArgBOffset) {  integers.push_back(args.b_offset); }
    else if (o == kArgCOffset) {  integers.push_back(args.c_offset); }
  }
  auto strings = std::vector<std::string>{};
  for (auto &o: options_) {
    if      (o == kArgAlpha) {    strings.push_back(ToString(args.alpha)); }
    else if (o == kArgBeta) {     strings.push_back(ToString(args.beta)); }
  }

  // Computes the GFLOPS and GB/s metrics
  auto flops = get_flops_(args);
  auto bytes = get_bytes_(args);
  auto gflops_clblast = (ms_clblast != 0.0) ? (flops*1e-6)/ms_clblast : 0;
  auto gflops_clblas = (ms_clblas != 0.0) ? (flops*1e-6)/ms_clblas: 0;
  auto gbs_clblast = (ms_clblast != 0.0) ? (bytes*1e-6)/ms_clblast : 0;
  auto gbs_clblas = (ms_clblas != 0.0) ? (bytes*1e-6)/ms_clblas: 0;

  // Outputs the argument values
  for (auto &argument: integers) {
    if (!args.no_abbrv && argument >= 1024*1024 && IsMultiple(argument, 1024*1024)) {
      fprintf(stdout, "%8luM;", argument/(1024*1024));
    }
    else if (!args.no_abbrv && argument >= 1024 && IsMultiple(argument, 1024)) {
      fprintf(stdout, "%8luK;", argument/1024);
    }
    else {
      fprintf(stdout, "%9lu;", argument);
    }
  }
  for (auto &argument: strings) {
    fprintf(stdout, "%9s;", argument.c_str());
  }

  // Outputs the performance numbers
  fprintf(stdout, "%9.2lf;%9.1lf;%9.1lf;%9.2lf;%9.1lf;%9.1lf\n",
          ms_clblast, gflops_clblast, gbs_clblast,
          ms_clblas, gflops_clblas, gbs_clblas);
}

// =================================================================================================

// Compiles the templated class
template class Client<float>;
template class Client<double>;
template class Client<float2>;
template class Client<double2>;

// =================================================================================================
} // namespace clblast
