
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemv command-line interface tester.
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
void PerformanceXgemv(const Arguments<T> &args,
                      const Buffer &a_mat, const Buffer &x_vec, const Buffer &y_vec,
                      CommandQueue &queue) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [&args, &a_mat, &x_vec, &y_vec, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Gemv(args.layout, args.a_transpose, args.m, args.n, args.alpha,
                       a_mat(), args.a_offset, args.a_ld,
                       x_vec(), args.x_offset, args.x_inc, args.beta,
                       y_vec(), args.y_offset, args.y_inc,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    if (status != StatusCode::kSuccess) {
      throw std::runtime_error("CLBlast error: "+ToString(static_cast<int>(status)));
    }
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [&args, &a_mat, &x_vec, &y_vec, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXgemv(static_cast<clblasOrder>(args.layout),
                              static_cast<clblasTranspose>(args.a_transpose),
                              args.m, args.n, args.alpha,
                              a_mat(), args.a_offset, args.a_ld,
                              x_vec(), args.x_offset, args.x_inc, args.beta,
                              y_vec(), args.y_offset, args.y_inc,
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
  const auto flops = 2 * args.m * args.n;
  const auto bytes = (args.m*args.n + 2*args.m + args.n) * sizeof(T);
  const auto output_ints = std::vector<size_t>{args.m, args.n,
                                               static_cast<size_t>(args.layout),
                                               static_cast<size_t>(args.a_transpose),
                                               args.a_ld, args.x_inc, args.y_inc,
                                               args.a_offset, args.x_offset, args.y_offset};
  const auto output_strings = std::vector<std::string>{ToString(args.alpha),
                                                       ToString(args.beta)};
  PrintTableRow(output_ints, output_strings, args.no_abbrv,
                ms_clblast, ms_clblas, flops, bytes);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void ClientXgemv(int argc, char *argv[]) {
  const auto o = std::vector<std::string>{kArgM, kArgN, kArgLayout, kArgATransp,
                                          kArgALeadDim, kArgXInc, kArgYInc,
                                          kArgAOffset, kArgXOffset, kArgYOffset,
                                          kArgAlpha, kArgBeta};
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: ClientAXY<float>(argc, argv, PerformanceXgemv<float>, o); break;
    case Precision::kDouble: ClientAXY<double>(argc, argv, PerformanceXgemv<double>, o); break;
    case Precision::kComplexSingle: ClientAXY<float2>(argc, argv, PerformanceXgemv<float2>, o); break;
    case Precision::kComplexDouble: ClientAXY<double2>(argc, argv, PerformanceXgemv<double2>, o); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::ClientXgemv(argc, argv);
  return 0;
}

// =================================================================================================
