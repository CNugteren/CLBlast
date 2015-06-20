
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xaxpy routine. It is based on the TestXY class.
//
// =================================================================================================

#include "wrapper_clblas.h"
#include "correctness/testxy.h"

namespace clblast {
// =================================================================================================

// The correctness tester, containing the function calls to CLBlast and to clBLAS for comparison.
template <typename T>
void XaxpyTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [](const Arguments<T> &args,
                           const Buffer &x_vec, const Buffer &y_vec,
                           CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    return Axpy(args.n, args.alpha,
                x_vec(), args.x_offset, args.x_inc,
                y_vec(), args.y_offset, args.y_inc,
                &queue_plain, &event);
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [](const Arguments<T> &args,
                          const Buffer &x_vec, const Buffer &y_vec,
                          CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXaxpy(args.n, args.alpha,
                              x_vec(), args.x_offset, args.x_inc,
                              y_vec(), args.y_offset, args.y_inc,
                              1, &queue_plain, 0, nullptr, &event);
    return static_cast<StatusCode>(status);
  };

  // Initializes the arguments relevant for this routine
  auto args = Arguments<T>{};
  const auto options = std::vector<std::string>{kArgN, kArgXInc, kArgYInc,
                                                kArgXOffset, kArgYOffset, kArgAlpha};

  // Creates a tester
  TestXY<T> tester{argc, argv, silent, name, options, clblast_lambda, clblas_lambda};

  // Runs the tests
  const auto case_name = "default";
  tester.TestRegular(args, case_name);
  tester.TestInvalidBufferSizes(args, case_name);
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::XaxpyTest<float>(argc, argv, false, "SAXPY");
  clblast::XaxpyTest<double>(argc, argv, true, "DAXPY");
  clblast::XaxpyTest<clblast::float2>(argc, argv, true, "CAXPY");
  clblast::XaxpyTest<clblast::double2>(argc, argv, true, "ZAXPY");
  return 0;
}

// =================================================================================================
