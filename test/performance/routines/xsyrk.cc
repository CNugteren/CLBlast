
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyrk command-line interface tester.
//
// =================================================================================================

#include <string>
#include <vector>
#include <exception>

#include "wrapper_clblas.h"
#include "performance/client.h"

namespace clblast {
// =================================================================================================

// The client, used for performance testing. It contains the function calls to CLBlast and to other
// libraries to compare against.
template <typename T>
void PerformanceXsyrk(const Arguments<T> &args,
                      const Buffer &a_mat, const Buffer &c_mat,
                      CommandQueue &queue) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [&args, &a_mat, &c_mat, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Syrk(args.layout, args.triangle, args.a_transpose,
                       args.n, args.k,
                       args.alpha,
                       a_mat(), args.a_offset, args.a_ld,
                       args.beta,
                       c_mat(), args.c_offset, args.c_ld,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    if (status != StatusCode::kSuccess) {
      throw std::runtime_error("CLBlast error: "+ToString(static_cast<int>(status)));
    }
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [&args, &a_mat, &c_mat, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXsyrk(static_cast<clblasOrder>(args.layout),
                              static_cast<clblasUplo>(args.triangle),
                              static_cast<clblasTranspose>(args.a_transpose),
                              args.n, args.k,
                              args.alpha,
                              a_mat(), args.a_offset, args.a_ld,
                              args.beta,
                              c_mat(), args.c_offset, args.c_ld,
                              1, &queue_plain, 0, nullptr, &event);
    clWaitForEvents(1, &event);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("clBLAS error: "+ToString(static_cast<int>(status)));
    }
  };

  // Runs the routines and collect the timings
  auto ms_clblast = TimedExecution(args.num_runs, clblast_lambda);
  auto ms_clblas = TimedExecution(args.num_runs, clblas_lambda);

  // Prints the performance of both libraries
  const auto flops = args.n * args.n * args.k;
  const auto bytes = (args.n*args.k + args.n*args.n) * sizeof(T);
  const auto output_ints = std::vector<size_t>{args.n, args.k,
                                               static_cast<size_t>(args.layout),
                                               static_cast<size_t>(args.triangle),
                                               static_cast<size_t>(args.a_transpose),
                                               args.a_ld, args.c_ld,
                                               args.a_offset, args.c_offset};
  const auto output_strings = std::vector<std::string>{ToString(args.alpha),
                                                       ToString(args.beta)};
  PrintTableRow(output_ints, output_strings, args.no_abbrv,
                ms_clblast, ms_clblas, flops, bytes);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void ClientXsyrk(int argc, char *argv[]) {
  const auto o = std::vector<std::string>{kArgN, kArgK,
                                          kArgLayout, kArgTriangle, kArgATransp,
                                          kArgALeadDim, kArgCLeadDim,
                                          kArgAOffset, kArgCOffset,
                                          kArgAlpha, kArgBeta};
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: ClientAC<float>(argc, argv, PerformanceXsyrk<float>, o); break;
    case Precision::kDouble: ClientAC<double>(argc, argv, PerformanceXsyrk<double>, o); break;
    case Precision::kComplexSingle: ClientAC<float2>(argc, argv, PerformanceXsyrk<float2>, o); break;
    case Precision::kComplexDouble: ClientAC<double2>(argc, argv, PerformanceXsyrk<double2>, o); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::ClientXsyrk(argc, argv);
  return 0;
}

// =================================================================================================
