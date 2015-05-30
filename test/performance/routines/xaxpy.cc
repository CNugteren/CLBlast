
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xaxpy command-line interface tester.
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
void PerformanceXaxpy(const Arguments<T> &args,
                      const Buffer &x_vec, const Buffer &y_vec,
                      CommandQueue &queue) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [&args, &x_vec, &y_vec, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = Axpy(args.n, args.alpha,
                       x_vec(), args.x_offset, args.x_inc,
                       y_vec(), args.y_offset, args.y_inc,
                       &queue_plain, &event);
    clWaitForEvents(1, &event);
    if (status != StatusCode::kSuccess) {
      throw std::runtime_error("CLBlast error: "+ToString(static_cast<int>(status)));
    }
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [&args, &x_vec, &y_vec, &queue]() {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXaxpy(args.n, args.alpha,
                              x_vec(), args.x_offset, args.x_inc,
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
  const auto flops = 2 * args.n;
  const auto bytes = (3 * args.n) * sizeof(T);
  const auto output_ints = std::vector<size_t>{args.n, args.x_inc, args.y_inc,
                                               args.x_offset, args.y_offset};
  const auto output_strings = std::vector<std::string>{ToString(args.alpha)};
  PrintTableRow(output_ints, output_strings, args.no_abbrv,
                ms_clblast, ms_clblas, flops, bytes);
}

// =================================================================================================

// Main function which calls the common client code with the routine-specific function as argument.
void ClientXaxpy(int argc, char *argv[]) {
  const auto o = std::vector<std::string>{kArgN, kArgXInc, kArgYInc,
                                          kArgXOffset, kArgYOffset, kArgAlpha};
  switch(GetPrecision(argc, argv)) {
    case Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case Precision::kSingle: ClientXY<float>(argc, argv, PerformanceXaxpy<float>, o); break;
    case Precision::kDouble: ClientXY<double>(argc, argv, PerformanceXaxpy<double>, o); break;
    case Precision::kComplexSingle: ClientXY<float2>(argc, argv, PerformanceXaxpy<float2>, o); break;
    case Precision::kComplexDouble: ClientXY<double2>(argc, argv, PerformanceXaxpy<double2>, o); break;
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::ClientXaxpy(argc, argv);
  return 0;
}

// =================================================================================================
