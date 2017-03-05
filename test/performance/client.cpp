
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

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <chrono>
#include <random>

#include "utilities/utilities.hpp"
#include "test/performance/client.hpp"

namespace clblast {
// =================================================================================================

// Constructor
template <typename T, typename U>
Client<T,U>::Client(const Routine run_routine,
                    const Routine run_reference1, const Routine run_reference2,
                    const std::vector<std::string> &options,
                    const GetMetric get_flops, const GetMetric get_bytes):
  run_routine_(run_routine),
  run_reference1_(run_reference1),
  run_reference2_(run_reference2),
  options_(options),
  get_flops_(get_flops),
  get_bytes_(get_bytes) {
}

// =================================================================================================

// Parses all arguments available for the CLBlast client testers. Some arguments might not be
// applicable, but are searched for anyway to be able to create one common argument parser. All
// arguments have a default value in case they are not found.
template <typename T, typename U>
Arguments<U> Client<T,U>::ParseArguments(int argc, char *argv[], const size_t level,
                                         const GetMetric default_a_ld,
                                         const GetMetric default_b_ld,
                                         const GetMetric default_c_ld) {
  const auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto args = Arguments<U>{};
  auto help = std::string{"\n* Options given/available:\n"};

  // These are the options which are not for every client: they are optional
  for (auto &o: options_) {

    // Data-sizes
    if (o == kArgM)  { args.m   = GetArgument(command_line_args, help, kArgM, size_t{512}); }
    if (o == kArgN)  { args.n   = GetArgument(command_line_args, help, kArgN, size_t{512}); }
    if (o == kArgK)  { args.k   = GetArgument(command_line_args, help, kArgK, size_t{512}); }
    if (o == kArgKU) { args.ku  = GetArgument(command_line_args, help, kArgKU, size_t{128}); }
    if (o == kArgKL) { args.kl  = GetArgument(command_line_args, help, kArgKL, size_t{128}); }

    // Data-layouts
    if (o == kArgLayout)   { args.layout      = GetArgument(command_line_args, help, kArgLayout, Layout::kRowMajor); }
    if (o == kArgATransp)  { args.a_transpose = GetArgument(command_line_args, help, kArgATransp, Transpose::kNo); }
    if (o == kArgBTransp)  { args.b_transpose = GetArgument(command_line_args, help, kArgBTransp, Transpose::kNo); }
    if (o == kArgSide)     { args.side        = GetArgument(command_line_args, help, kArgSide, Side::kLeft); }
    if (o == kArgTriangle) { args.triangle    = GetArgument(command_line_args, help, kArgTriangle, Triangle::kUpper); }
    if (o == kArgDiagonal) { args.diagonal    = GetArgument(command_line_args, help, kArgDiagonal, Diagonal::kUnit); }

    // Vector arguments
    if (o == kArgXInc)    { args.x_inc    = GetArgument(command_line_args, help, kArgXInc, size_t{1}); }
    if (o == kArgYInc)    { args.y_inc    = GetArgument(command_line_args, help, kArgYInc, size_t{1}); }
    if (o == kArgXOffset) { args.x_offset = GetArgument(command_line_args, help, kArgXOffset, size_t{0}); }
    if (o == kArgYOffset) { args.y_offset = GetArgument(command_line_args, help, kArgYOffset, size_t{0}); }

    // Matrix arguments
    if (o == kArgALeadDim) { args.a_ld     = GetArgument(command_line_args, help, kArgALeadDim, default_a_ld(args)); }
    if (o == kArgBLeadDim) { args.b_ld     = GetArgument(command_line_args, help, kArgBLeadDim, default_b_ld(args)); }
    if (o == kArgCLeadDim) { args.c_ld     = GetArgument(command_line_args, help, kArgCLeadDim, default_c_ld(args)); }
    if (o == kArgAOffset)  { args.a_offset = GetArgument(command_line_args, help, kArgAOffset, size_t{0}); }
    if (o == kArgBOffset)  { args.b_offset = GetArgument(command_line_args, help, kArgBOffset, size_t{0}); }
    if (o == kArgCOffset)  { args.c_offset = GetArgument(command_line_args, help, kArgCOffset, size_t{0}); }
    if (o == kArgAPOffset) { args.ap_offset= GetArgument(command_line_args, help, kArgAPOffset, size_t{0}); }

    // Scalar result arguments
    if (o == kArgDotOffset)  { args.dot_offset = GetArgument(command_line_args, help, kArgDotOffset, size_t{0}); }
    if (o == kArgNrm2Offset)  { args.nrm2_offset = GetArgument(command_line_args, help, kArgNrm2Offset, size_t{0}); }
    if (o == kArgAsumOffset)  { args.asum_offset = GetArgument(command_line_args, help, kArgAsumOffset, size_t{0}); }
    if (o == kArgImaxOffset)  { args.imax_offset = GetArgument(command_line_args, help, kArgImaxOffset, size_t{0}); }

    // Scalar values 
    if (o == kArgAlpha) { args.alpha = GetArgument(command_line_args, help, kArgAlpha, GetScalar<U>()); }
    if (o == kArgBeta)  { args.beta  = GetArgument(command_line_args, help, kArgBeta, GetScalar<U>()); }
  }

  // These are the options common to all routines
  args.platform_id    = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  args.device_id      = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  args.precision      = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  #ifdef CLBLAST_REF_CLBLAS
    args.compare_clblas = GetArgument(command_line_args, help, kArgCompareclblas, 1);
  #else
    args.compare_clblas = 0;
  #endif
  #ifdef CLBLAST_REF_CBLAS
    args.compare_cblas  = GetArgument(command_line_args, help, kArgComparecblas, 1);
  #else
    args.compare_cblas = 0;
  #endif
  args.step           = GetArgument(command_line_args, help, kArgStepSize, size_t{1});
  args.num_steps      = GetArgument(command_line_args, help, kArgNumSteps, size_t{0});
  args.num_runs       = GetArgument(command_line_args, help, kArgNumRuns, size_t{10});
  args.print_help     = CheckArgument(command_line_args, help, kArgHelp);
  args.silent         = CheckArgument(command_line_args, help, kArgQuiet);
  args.no_abbrv       = CheckArgument(command_line_args, help, kArgNoAbbreviations);
  warm_up_            = CheckArgument(command_line_args, help, kArgWarmUp);

  // Prints the chosen (or defaulted) arguments to screen. This also serves as the help message,
  // which is thus always displayed (unless silence is specified).
  if (!args.silent) { fprintf(stdout, "%s\n", help.c_str()); }

  // Comparison against a non-BLAS routine is not supported
  if (level == 4) { // level-4 == level-X
    if (args.compare_clblas != 0 || args.compare_cblas != 0) {
      if (!args.silent) {
        fprintf(stdout, "* Disabling clBLAS and CPU BLAS comparisons for this non-BLAS routine\n\n");
      }
    }
    args.compare_clblas = 0;
    args.compare_cblas = 0;
  }

  // Comparison against clBLAS or a CPU BLAS library is not supported in case of half-precision
  if (args.precision == Precision::kHalf) {
    if (args.compare_clblas != 0 || args.compare_cblas != 0) {
      if (!args.silent) {
        fprintf(stdout, "* Disabling clBLAS and CPU BLAS comparisons for half-precision\n\n");
      }
    }
    args.compare_clblas = 0;
    args.compare_cblas = 0;
  }

  // Returns the arguments
  return args;
}

// =================================================================================================

// This is main performance tester
template <typename T, typename U>
void Client<T,U>::PerformanceTest(Arguments<U> &args, const SetMetric set_sizes) {

  // Prints the header of the output table
  PrintTableHeader(args);

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, args.device_id);
  auto context = Context(device);
  auto queue = Queue(context, device);
  #ifdef CLBLAST_REF_CLBLAS
    if (args.compare_clblas) { clblasSetup(); }
  #endif

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
    std::vector<T> ap_source(args.ap_size);
    std::vector<T> scalar_source(args.scalar_size);
    std::mt19937 mt(kSeed);
    std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
    PopulateVector(x_source, mt, dist);
    PopulateVector(y_source, mt, dist);
    PopulateVector(a_source, mt, dist);
    PopulateVector(b_source, mt, dist);
    PopulateVector(c_source, mt, dist);
    PopulateVector(ap_source, mt, dist);
    PopulateVector(scalar_source, mt, dist);

    // Creates the matrices on the device
    auto x_vec = Buffer<T>(context, args.x_size);
    auto y_vec = Buffer<T>(context, args.y_size);
    auto a_mat = Buffer<T>(context, args.a_size);
    auto b_mat = Buffer<T>(context, args.b_size);
    auto c_mat = Buffer<T>(context, args.c_size);
    auto ap_mat = Buffer<T>(context, args.ap_size);
    auto scalar = Buffer<T>(context, args.scalar_size);
    x_vec.Write(queue, args.x_size, x_source);
    y_vec.Write(queue, args.y_size, y_source);
    a_mat.Write(queue, args.a_size, a_source);
    b_mat.Write(queue, args.b_size, b_source);
    c_mat.Write(queue, args.c_size, c_source);
    ap_mat.Write(queue, args.ap_size, ap_source);
    scalar.Write(queue, args.scalar_size, scalar_source);
    auto buffers = Buffers<T>{x_vec, y_vec, a_mat, b_mat, c_mat, ap_mat, scalar};

    // Runs the routines and collects the timings
    auto timings = std::vector<std::pair<std::string, double>>();
    auto ms_clblast = TimedExecution(args.num_runs, args, buffers, queue, run_routine_, "CLBlast");
    timings.push_back(std::pair<std::string, double>("CLBlast", ms_clblast));
    if (args.compare_clblas) {
      auto ms_clblas = TimedExecution(args.num_runs, args, buffers, queue, run_reference1_, "clBLAS");
      timings.push_back(std::pair<std::string, double>("clBLAS", ms_clblas));
    }
    if (args.compare_cblas) {
      auto ms_cblas = TimedExecution(args.num_runs, args, buffers, queue, run_reference2_, "CPU BLAS");
      timings.push_back(std::pair<std::string, double>("CPU BLAS", ms_cblas));
    }

    // Prints the performance of the tested libraries
    PrintTableRow(args, timings);

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
  #ifdef CLBLAST_REF_CLBLAS
    if (args.compare_clblas) { clblasTeardown(); }
  #endif
}

// =================================================================================================

// Creates a vector of timing results, filled with execution times of the 'main computation'. The
// timing is performed using the milliseconds chrono functions. The function returns the minimum
// value found in the vector of timing results. The return value is in milliseconds.
template <typename T, typename U>
double Client<T,U>::TimedExecution(const size_t num_runs, const Arguments<U> &args,
                                   Buffers<T> &buffers, Queue &queue,
                                   Routine run_blas, const std::string &library_name) {
  auto status = StatusCode::kSuccess;

  // Do an optional warm-up to omit compilation times and initialisations from the measurements
  if (warm_up_) {
    try {
      status = run_blas(args, buffers, queue);
    } catch (...) { status = static_cast<StatusCode>(kUnknownError); }
    if (status != StatusCode::kSuccess) {
      throw std::runtime_error(library_name+" error: "+ToString(static_cast<int>(status)));
    }
  }

  // Start the timed part
  auto timings = std::vector<double>(num_runs);
  for (auto &timing: timings) {
    auto start_time = std::chrono::steady_clock::now();

    // Executes the main computation
    try {
      status = run_blas(args, buffers, queue);
    } catch (...) { status = static_cast<StatusCode>(kUnknownError); }
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
template <typename T, typename U>
void Client<T,U>::PrintTableHeader(const Arguments<U>& args) {

  // First line (optional)
  if (!args.silent) {
    for (auto i=size_t{0}; i<options_.size(); ++i) { fprintf(stdout, "%9s ", ""); }
    fprintf(stdout, " | <--       CLBlast       -->");
    if (args.compare_clblas) { fprintf(stdout, " | <--       clBLAS        -->"); }
    if (args.compare_cblas) { fprintf(stdout, " | <--      CPU BLAS       -->"); }
    fprintf(stdout, " |\n");
  }

  // Second line
  for (auto &option: options_) { fprintf(stdout, "%9s;", option.c_str()); }
  fprintf(stdout, "%9s;%9s;%9s", "ms_1", "GFLOPS_1", "GBs_1");
  if (args.compare_clblas) { fprintf(stdout, ";%9s;%9s;%9s", "ms_2", "GFLOPS_2", "GBs_2"); }
  if (args.compare_cblas) { fprintf(stdout, ";%9s;%9s;%9s", "ms_3", "GFLOPS_3", "GBs_3"); }
  fprintf(stdout, "\n");
}

// Print a performance-result row
template <typename T, typename U>
void Client<T,U>::PrintTableRow(const Arguments<U>& args,
                                const std::vector<std::pair<std::string, double>>& timings) {

  // Creates a vector of relevant variables
  auto integers = std::vector<size_t>{};
  for (auto &o: options_) {
    if      (o == kArgM) {        integers.push_back(args.m); }
    else if (o == kArgN) {        integers.push_back(args.n); }
    else if (o == kArgK) {        integers.push_back(args.k); }
    else if (o == kArgKU) {       integers.push_back(args.ku); }
    else if (o == kArgKL) {       integers.push_back(args.kl); }
    else if (o == kArgLayout) {   integers.push_back(static_cast<size_t>(args.layout)); }
    else if (o == kArgSide) {     integers.push_back(static_cast<size_t>(args.side)); }
    else if (o == kArgTriangle) { integers.push_back(static_cast<size_t>(args.triangle)); }
    else if (o == kArgATransp) {  integers.push_back(static_cast<size_t>(args.a_transpose)); }
    else if (o == kArgBTransp) {  integers.push_back(static_cast<size_t>(args.b_transpose)); }
    else if (o == kArgDiagonal) { integers.push_back(static_cast<size_t>(args.diagonal)); }
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
    else if (o == kArgAPOffset) { integers.push_back(args.ap_offset); }
    else if (o == kArgDotOffset) {integers.push_back(args.dot_offset); }
    else if (o == kArgNrm2Offset){integers.push_back(args.nrm2_offset); }
    else if (o == kArgAsumOffset){integers.push_back(args.asum_offset); }
    else if (o == kArgImaxOffset){integers.push_back(args.imax_offset); }
  }
  auto strings = std::vector<std::string>{};
  for (auto &o: options_) {
    if      (o == kArgAlpha) {    strings.push_back(ToString(args.alpha)); }
    else if (o == kArgBeta) {     strings.push_back(ToString(args.beta)); }
  }

  // Outputs the argument values
  for (auto &argument: integers) {
    if (!args.no_abbrv && argument >= 1024*1024 && IsMultiple(argument, 1024*1024)) {
      fprintf(stdout, "%8zuM;", argument/(1024*1024));
    }
    else if (!args.no_abbrv && argument >= 1024 && IsMultiple(argument, 1024)) {
      fprintf(stdout, "%8zuK;", argument/1024);
    }
    else {
      fprintf(stdout, "%9zu;", argument);
    }
  }
  for (auto &argument: strings) {
    fprintf(stdout, "%9s;", argument.c_str());
  }

  // Loops over all tested libraries
  for (const auto& timing : timings) {

    // Computes the GFLOPS and GB/s metrics
    auto flops = get_flops_(args);
    auto bytes = get_bytes_(args);
    auto gflops = (timing.second != 0.0) ? (flops*1e-6)/timing.second : 0;
    auto gbs = (timing.second != 0.0) ? (bytes*1e-6)/timing.second : 0;

    // Outputs the performance numbers
    if (timing.first != "CLBlast") { fprintf(stdout, ";"); }
    fprintf(stdout, "%9.2lf;%9.1lf;%9.1lf", timing.second, gflops, gbs);
  }
  fprintf(stdout, "\n");
}

// =================================================================================================

// Compiles the templated class
template class Client<half,half>;
template class Client<float,float>;
template class Client<double,double>;
template class Client<float2,float2>;
template class Client<double2,double2>;
template class Client<float2,float>;
template class Client<double2,double>;

// =================================================================================================
} // namespace clblast
