
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xsyrk routine. It is based on the TestAC class.
//
// =================================================================================================

#include "wrapper_clblas.h"
#include "correctness/testac.h"

namespace clblast {
// =================================================================================================

// The correctness tester, containing the function calls to CLBlast and to clBLAS for comparison.
template <typename T>
void XsyrkTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [](const Arguments<T> &args,
                           const Buffer &a_mat, const Buffer &c_mat,
                           CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    return Syrk(args.layout, args.triangle, args.a_transpose,
                args.n, args.k,
                args.alpha,
                a_mat(), args.a_offset, args.a_ld,
                args.beta,
                c_mat(), args.c_offset, args.c_ld,
                &queue_plain, &event);
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [](const Arguments<T> &args,
                          const Buffer &a_mat, const Buffer &c_mat,
                          CommandQueue &queue) -> StatusCode {
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
    return static_cast<StatusCode>(status);
  };

  // Initializes the arguments relevant for this routine
  auto args = Arguments<T>{};
  const auto options = std::vector<std::string>{kArgN, kArgK, kArgLayout,
                                                kArgTriangle, kArgATransp,
                                                kArgALeadDim, kArgCLeadDim,
                                                kArgAOffset, kArgCOffset};

  // Creates a tester
  TestAC<T> tester{argc, argv, silent, name, options, clblast_lambda, clblas_lambda};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) {
    args.layout = layout;
    for (auto &triangle: {Triangle::kUpper, Triangle::kLower}) {
      args.triangle = triangle;
      for (auto &a_transpose: {Transpose::kNo, Transpose::kYes}) { // No conjugate here since it is
        args.a_transpose = a_transpose;                            // not supported by clBLAS
        const auto case_name = ToString(layout)+" "+ToString(triangle)+" "+ToString(a_transpose);

        // Runs the tests
        tester.TestRegular(args, case_name);
        tester.TestInvalidBufferSizes(args, case_name);
      }
    }
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::XsyrkTest<float>(argc, argv, false, "SSYRK");
  clblast::XsyrkTest<double>(argc, argv, true, "DSYRK");
  clblast::XsyrkTest<clblast::float2>(argc, argv, true, "CSYRK");
  clblast::XsyrkTest<clblast::double2>(argc, argv, true, "ZSYRK");
  return 0;
}

// =================================================================================================
