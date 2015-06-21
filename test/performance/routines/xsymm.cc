
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsymm command-line interface tester.
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
void PerformanceXsymm(const Arguments<T> &args,
                      const Buffer &a_mat, const Buffer &b_mat, const Buffer &c_mat,
                      CommandQueue &queue) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [&args, &a_mat, &b_mat, &c_mat, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Symm(args.layout, args.side, args.triangle,
                       args.m, args.n,
                       args.alpha,
                       a_mat(), args.a_offset, args.a_ld,
                       b_mat(), args.b_offset, args.b_ld,
                       args.beta,
                       c_mat(), args.c_offset, args.c_ld,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    if (status != StatusCode::kSuccess) {
      throw std::runtime_error("CLBlast error: "+ToString(static_cast<int>(status)));
    }
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [&args, &a_mat, &b_mat, &c_mat, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXsymm(static_cast<clblasOrder>(args.layout),
                              static_cast<clblasSide>(args.side),
                              static_cast<clblasUplo>(args.triangle),
                              args.m, args.n,
                              args.alpha,
                              a_mat(), args.a_offset, args.a_ld,
                              b_mat(), args.b_offset, args.b_ld,
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
  const auto flops = 2 * args.m * args.n * args.m;
  const auto bytes = (args.m*args.m + args.m*args.n + args.m*args.n) * sizeof(T);
  const auto output_ints = std::vector<size_t>{args.m, args.n,
                                               static_cast<size_t>(args.layout),
                                               static_cast<size_t>(args.triangle),
                                               static_cast<size_t>(args.side),
                                               args.a_ld, args.b_ld, args.c_ld,
                                               args.a_offset, args.b_offset, args.c_offset};
  const auto output_strings = std::vector<std::string>{ToString(args.alpha),
                                                       ToString(args.beta)};
  PrintTableRow(output_ints, output_strings, args.no_abbrv,
                ms_clblast, ms_clblas, flops, bytes);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void ClientXsymm(int argc, char *argv[]) {
  const auto o = std::vector<std::string>{kArgM, kArgN, kArgLayout,
                                          kArgTriangle, kArgSide,
                                          kArgALeadDim, kArgBLeadDim, kArgCLeadDim,
                                          kArgAOffset, kArgBOffset, kArgCOffset,
                                          kArgAlpha, kArgBeta};
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: ClientABC<float>(argc, argv, PerformanceXsymm<float>, o); break;
    case Precision::kDouble: ClientABC<double>(argc, argv, PerformanceXsymm<double>, o); break;
    case Precision::kComplexSingle: ClientABC<float2>(argc, argv, PerformanceXsymm<float2>, o); break;
    case Precision::kComplexDouble: ClientABC<double2>(argc, argv, PerformanceXsymm<double2>, o); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::ClientXsymm(argc, argv);
  return 0;
}

// =================================================================================================
