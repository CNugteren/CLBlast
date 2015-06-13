
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xgemv routine. It is based on the TestAXY class.
//
// =================================================================================================

#include "wrapper_clblas.h"
#include "correctness/testaxy.h"

namespace clblast {
// =================================================================================================

// The correctness tester, containing the function calls to CLBlast and to clBLAS for comparison.
template <typename T>
void XgemvTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [](const Arguments<T> &args,
                           const Buffer &a_mat, const Buffer &x_vec, const Buffer &y_vec,
                           CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    return Gemv(args.layout, args.a_transpose, args.m, args.n, args.alpha,
                a_mat(), args.a_offset, args.a_ld,
                x_vec(), args.x_offset, args.x_inc, args.beta,
                y_vec(), args.y_offset, args.y_inc,
                &queue_plain, &event);
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [](const Arguments<T> &args,
                          const Buffer &a_mat, const Buffer &x_vec, const Buffer &y_vec,
                          CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXgemv(static_cast<clblasOrder>(args.layout),
                              static_cast<clblasTranspose>(args.a_transpose),
                              args.m, args.n, args.alpha,
                              a_mat(), args.a_offset, args.a_ld,
                              x_vec(), args.x_offset, args.x_inc, args.beta,
                              y_vec(), args.y_offset, args.y_inc,
                              1, &queue_plain, 0, nullptr, &event);
    return static_cast<StatusCode>(status);
  };

  // Selects the platform and device on which to test (command-line options)
  auto help = std::string{"Options given/available:\n"};
  const auto platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  const auto device_id = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes the other arguments relevant for this routine
  auto args = Arguments<T>{};
  const auto options = std::vector<std::string>{kArgM, kArgN, kArgLayout, kArgATransp,
                                                kArgALeadDim, kArgXInc, kArgYInc,
                                                kArgAOffset, kArgXOffset, kArgYOffset};

  // Creates a tester
  TestAXY<T> tester{platform_id, device_id, name, options, clblast_lambda, clblas_lambda};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: {Layout::kRowMajor, Layout::kColMajor}) {
    args.layout = layout;
    for (auto &a_transpose: {Transpose::kNo, Transpose::kYes}) {
      args.a_transpose = a_transpose;
      const auto case_name = ToString(layout)+" "+ToString(a_transpose);

      // Runs the tests
      tester.TestRegular(args, case_name);
      tester.TestInvalidBufferSizes(args, case_name);
    }
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::XgemvTest<float>(argc, argv, false, "SGEMV");
  //clblast::XgemvTest<double>(argc, argv, true, "DGEMV");
  //clblast::XgemvTest<clblast::float2>(argc, argv, true, "CGEMV");
  //clblast::XgemvTest<clblast::double2>(argc, argv, true, "ZGEMV");
  return 0;
}

// =================================================================================================
