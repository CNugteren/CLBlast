
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
#include <tuning/tuning.hpp>

#include "utilities/utilities.hpp"
#include "test/performance/client.hpp"

namespace clblast {
// =================================================================================================

template <typename T, typename U> const int Client<T,U>::kSeed = 42; // fixed seed for reproducibility

// Constructor
template <typename T, typename U>
Client<T,U>::Client(const Routine run_routine,
                    const Reference1 run_reference1, const Reference2 run_reference2,
                    const Reference3 run_reference3, const std::vector<std::string> &options,
                    const std::vector<std::string> &buffers_in,
                    const std::vector<std::string> &buffers_out,
                    const GetMetric get_flops, const GetMetric get_bytes):
  run_routine_(run_routine),
  run_reference1_(run_reference1),
  run_reference2_(run_reference2),
  run_reference3_(run_reference3),
  options_(options),
  buffers_in_(buffers_in),
  buffers_out_(buffers_out),
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
    if (o == kArgDotOffset)   { args.dot_offset = GetArgument(command_line_args, help, kArgDotOffset, size_t{0}); }
    if (o == kArgNrm2Offset)  { args.nrm2_offset = GetArgument(command_line_args, help, kArgNrm2Offset, size_t{0}); }
    if (o == kArgAsumOffset)  { args.asum_offset = GetArgument(command_line_args, help, kArgAsumOffset, size_t{0}); }
    if (o == kArgImaxOffset)  { args.imax_offset = GetArgument(command_line_args, help, kArgImaxOffset, size_t{0}); }

    // Batch arguments
    if (o == kArgBatchCount) { args.batch_count = GetArgument(command_line_args, help, kArgBatchCount, size_t{1}); }

    // Scalar values 
    if (o == kArgAlpha) { args.alpha = GetArgument(command_line_args, help, kArgAlpha, GetScalar<U>()); }
    if (o == kArgBeta)  { args.beta  = GetArgument(command_line_args, help, kArgBeta, GetScalar<U>()); }

    // Arguments for im2col and convgemm
    if (o == kArgKernelMode){ args.kernel_mode = GetArgument(command_line_args, help, kArgKernelMode, KernelMode::kConvolution); }
    if (o == kArgChannels)  { args.channels = GetArgument(command_line_args, help, kArgChannels, size_t{64}); }
    if (o == kArgHeight)    { args.height = GetArgument(command_line_args, help, kArgHeight, size_t{64}); }
    if (o == kArgWidth)     { args.width = GetArgument(command_line_args, help, kArgWidth, size_t{64}); }
    if (o == kArgKernelH)   { args.kernel_h = GetArgument(command_line_args, help, kArgKernelH, size_t{3}); }
    if (o == kArgKernelW)   { args.kernel_w = GetArgument(command_line_args, help, kArgKernelW, size_t{3}); }
    if (o == kArgPadH)      { args.pad_h = GetArgument(command_line_args, help, kArgPadH, size_t{0}); }
    if (o == kArgPadW)      { args.pad_w = GetArgument(command_line_args, help, kArgPadW, size_t{0}); }
    if (o == kArgStrideH)   { args.stride_h = GetArgument(command_line_args, help, kArgStrideH, size_t{1}); }
    if (o == kArgStrideW)   { args.stride_w = GetArgument(command_line_args, help, kArgStrideW, size_t{1}); }
    if (o == kArgDilationH) { args.dilation_h = GetArgument(command_line_args, help, kArgDilationH, size_t{1}); }
    if (o == kArgDilationW) { args.dilation_w = GetArgument(command_line_args, help, kArgDilationW, size_t{1}); }
    if (o == kArgNumKernels){ args.num_kernels = GetArgument(command_line_args, help, kArgNumKernels, size_t{1}); }
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
  #ifdef CLBLAST_REF_CUBLAS
    args.compare_cublas  = GetArgument(command_line_args, help, kArgComparecublas, 1);
  #else
    args.compare_cublas = 0;
  #endif
  args.step           = GetArgument(command_line_args, help, kArgStepSize, size_t{1});
  args.num_steps      = GetArgument(command_line_args, help, kArgNumSteps, size_t{0});
  args.num_runs       = GetArgument(command_line_args, help, kArgNumRuns, size_t{10});
  args.print_help     = CheckArgument(command_line_args, help, kArgHelp);
  args.silent         = CheckArgument(command_line_args, help, kArgQuiet);
  args.no_abbrv       = CheckArgument(command_line_args, help, kArgNoAbbreviations);
  args.full_statistics= CheckArgument(command_line_args, help, kArgFullStatistics);
  warm_up_            = CheckArgument(command_line_args, help, kArgWarmUp);

  // Parse the optional JSON file name arguments
  const auto tuner_files_default = std::string{"<none>"};
  const auto tuner_files_string = GetArgument(command_line_args, help, kArgTunerFiles, tuner_files_default);
  if (tuner_files_string != tuner_files_default) {
    args.tuner_files = split(tuner_files_string, ',');
  }

  // Prints the chosen (or defaulted) arguments to screen. This also serves as the help message,
  // which is thus always displayed (unless silence is specified).
  if (!args.silent) { fprintf(stdout, "%s\n", help.c_str()); }

  // Comparison against a non-BLAS routine is not supported
  if (level == 4) { // level-4 == level-X
    if (args.compare_clblas != 0 || args.compare_cblas != 0 || args.compare_cublas != 0) {
      if (!args.silent) {
        fprintf(stdout, "* Disabling clBLAS/CBLAS/cuBLAS comparisons for this non-BLAS routine\n\n");
      }
    }
    args.compare_clblas = 0;
    args.compare_cblas = 0;
    args.compare_cublas = 0;
  }

  // Comparison against other BLAS libraries is not supported in case of half-precision
  if (args.precision == Precision::kHalf) {
    if (args.compare_clblas != 0 || args.compare_cblas != 0 || args.compare_cublas != 0) {
      if (!args.silent) {
        fprintf(stdout, "* Disabling clBLAS/CBLAS/cuBLAS comparisons for half-precision\n\n");
      }
    }
    args.compare_clblas = 0;
    args.compare_cblas = 0;
    args.compare_cublas = 0;
  }

  // Returns the arguments
  return args;
}

// =================================================================================================

// This is main performance tester
template <typename T, typename U>
void Client<T,U>::PerformanceTest(Arguments<U> &args, const SetMetric set_sizes) {

  // Initializes OpenCL and the libraries
  auto platform = Platform(args.platform_id);
  auto device = Device(platform, args.device_id);
  auto context = Context(device);
  auto queue = Queue(context, device);
  #ifdef CLBLAST_REF_CLBLAS
    if (args.compare_clblas) { clblasSetup(); }
  #endif
  #ifdef CLBLAST_REF_CUBLAS
    if (args.compare_cublas) { cublasSetup(args); }
  #endif

  // Optionally overrides parameters if tuner files are given (semicolon separated)
  OverrideParametersFromJSONFiles(args.tuner_files, device(), args.precision);

  // Prints the header of the output table
  PrintTableHeader(args);

  // Iterates over all "num_step" values jumping by "step" each time
  auto s = size_t{0};
  while(true) {

    // Sets the buffer sizes (routine-specific)
    set_sizes(args, queue);

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
    auto timings = std::vector<std::pair<std::string, TimeResult>>();
    auto time_clblast = TimedExecution(args.num_runs, args, buffers, queue, run_routine_, "CLBlast");
    timings.push_back(std::pair<std::string, TimeResult>("CLBlast", time_clblast));
    if (args.compare_clblas) {
      auto time_clblas = TimedExecution(args.num_runs, args, buffers, queue, run_reference1_, "clBLAS");
      timings.push_back(std::pair<std::string, TimeResult>("clBLAS", time_clblas));
    }
    if (args.compare_cblas) {
      auto buffers_host = BuffersHost<T>();
      DeviceToHost(args, buffers, buffers_host, queue, buffers_in_);
      auto time_cblas = TimedExecution(args.num_runs, args, buffers_host, queue, run_reference2_, "CPU BLAS");
      HostToDevice(args, buffers, buffers_host, queue, buffers_out_);
      timings.push_back(std::pair<std::string, TimeResult>("CPU BLAS", time_cblas));
    }
    if (args.compare_cublas) {
      auto buffers_host = BuffersHost<T>();
      auto buffers_cuda = BuffersCUDA<T>();
      DeviceToHost(args, buffers, buffers_host, queue, buffers_in_);
      HostToCUDA(args, buffers_cuda, buffers_host, buffers_in_);
      TimeResult time_cublas;
      try {
        time_cublas = TimedExecution(args.num_runs, args, buffers_cuda, queue, run_reference3_, "cuBLAS");
      } catch (std::runtime_error e) { }
      CUDAToHost(args, buffers_cuda, buffers_host, buffers_out_);
      HostToDevice(args, buffers, buffers_host, queue, buffers_out_);
      timings.push_back(std::pair<std::string, TimeResult>("cuBLAS", time_cublas));
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
  #ifdef CLBLAST_REF_CUBLAS
    if (args.compare_cublas) { cublasTeardown(args); }
  #endif
}

// =================================================================================================

// Creates a vector of timing results, filled with execution times of the 'main computation'. The
// timing is performed using the milliseconds chrono functions. The function returns the minimum
// value found in the vector of timing results. The return value is in milliseconds.
template <typename T, typename U>
template <typename BufferType, typename RoutineType>
typename Client<T,U>::TimeResult Client<T,U>::TimedExecution(const size_t num_runs, const Arguments<U> &args,
                                                             BufferType &buffers, Queue &queue,
                                                             RoutineType run_blas, const std::string &library_name) {
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

  // Compute statistics
  auto result = TimeResult();
  const auto sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  const auto mean = sum / timings.size();
  std::vector<double> diff(timings.size());
  std::transform(timings.begin(), timings.end(), diff.begin(), [mean](double x) { return x - mean; });
  const auto sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  result.mean = mean;
  result.standard_deviation = std::sqrt(sq_sum / timings.size());
  result.minimum = *std::min_element(timings.begin(), timings.end());
  result.maximum = *std::max_element(timings.begin(), timings.end());
  return result;
}

// =================================================================================================

// Prints the header of the performance table
template <typename T, typename U>
void Client<T,U>::PrintTableHeader(const Arguments<U>& args) {

  // First line (optional)
  if (!args.silent) {
    for (auto i=size_t{0}; i<options_.size(); ++i) { fprintf(stdout, "%9s ", ""); }
    if (args.full_statistics) {
      fprintf(stdout, " | <--            CLBlast            -->");
      if (args.compare_clblas) { fprintf(stdout, " | <--            clBLAS             -->"); }
      if (args.compare_cblas) { fprintf(stdout, " | <--           CPU BLAS            -->"); }
      if (args.compare_cublas) { fprintf(stdout, " | <--            cuBLAS             -->"); }
    }
    else {
      fprintf(stdout, " | <--       CLBlast       -->");
      if (args.compare_clblas) { fprintf(stdout, " | <--       clBLAS        -->"); }
      if (args.compare_cblas) { fprintf(stdout, " | <--      CPU BLAS       -->"); }
      if (args.compare_cublas) { fprintf(stdout, " | <--       cuBLAS        -->"); }
    }
    fprintf(stdout, " |\n");
  }

  // Second line
  for (auto &option: options_) { fprintf(stdout, "%9s;", option.c_str()); }
  if (args.full_statistics) {
    fprintf(stdout, "%9s;%9s;%9s;%9s", "min_ms_1", "max_ms_1", "mean_1", "stddev_1");
    if (args.compare_clblas) { fprintf(stdout, ";%9s;%9s;%9s;%9s", "min_ms_2", "max_ms_2", "mean_2", "stddev_2"); }
    if (args.compare_cblas) { fprintf(stdout, ";%9s;%9s;%9s;%9s", "min_ms_3", "max_ms_3", "mean_3", "stddev_3"); }
    if (args.compare_cublas) { fprintf(stdout, ";%9s;%9s;%9s;%9s", "min_ms_4", "max_ms_4", "mean_4", "stddev_4"); }
  }
  else {
    fprintf(stdout, "%9s;%9s;%9s", "ms_1", "GFLOPS_1", "GBs_1");
    if (args.compare_clblas) { fprintf(stdout, ";%9s;%9s;%9s", "ms_2", "GFLOPS_2", "GBs_2"); }
    if (args.compare_cblas) { fprintf(stdout, ";%9s;%9s;%9s", "ms_3", "GFLOPS_3", "GBs_3"); }
    if (args.compare_cublas) { fprintf(stdout, ";%9s;%9s;%9s", "ms_4", "GFLOPS_4", "GBs_4"); }
  }
  fprintf(stdout, "\n");
}

// Print a performance-result row
template <typename T, typename U>
void Client<T,U>::PrintTableRow(const Arguments<U>& args,
                                const std::vector<std::pair<std::string, TimeResult>>& timings) {

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
    else if (o == kArgBatchCount){integers.push_back(args.batch_count); }
    else if (o == kArgKernelMode){integers.push_back(static_cast<size_t>(args.kernel_mode)); }
    else if (o == kArgChannels)  {integers.push_back(args.channels); }
    else if (o == kArgHeight)    {integers.push_back(args.height); }
    else if (o == kArgWidth)     {integers.push_back(args.width); }
    else if (o == kArgKernelH)   {integers.push_back(args.kernel_h); }
    else if (o == kArgKernelW)   {integers.push_back(args.kernel_w); }
    else if (o == kArgPadH)      {integers.push_back(args.pad_h); }
    else if (o == kArgPadW)      {integers.push_back(args.pad_w); }
    else if (o == kArgStrideH)   {integers.push_back(args.stride_h); }
    else if (o == kArgStrideW)   {integers.push_back(args.stride_w); }
    else if (o == kArgDilationH) {integers.push_back(args.dilation_h); }
    else if (o == kArgDilationW) {integers.push_back(args.dilation_w); }
    else if (o == kArgNumKernels){integers.push_back(args.num_kernels); }
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
    const auto library_name = timing.first;
    const auto minimum_ms = timing.second.minimum;
    if (library_name != "CLBlast") { fprintf(stdout, ";"); }

    // Either output full statistics
    if (args.full_statistics) {
      const auto maximum_ms = timing.second.maximum;
      const auto mean_ms = timing.second.mean;
      const auto standard_deviation = timing.second.standard_deviation;
      fprintf(stdout, "%9.3lf;%9.3lf;%9.3lf;%9.3lf", minimum_ms, maximum_ms, mean_ms, standard_deviation);
    }

    // ... or outputs minimum time and the GFLOPS and GB/s metrics
    else {
      const auto flops = get_flops_(args);
      const auto bytes = get_bytes_(args);
      const auto gflops = (minimum_ms != 0.0) ? (flops*1e-6)/minimum_ms : 0;
      const auto gbs = (minimum_ms != 0.0) ? (bytes*1e-6)/minimum_ms : 0;
      fprintf(stdout, "%9.2lf;%9.1lf;%9.1lf", minimum_ms, gflops, gbs);
    }
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
