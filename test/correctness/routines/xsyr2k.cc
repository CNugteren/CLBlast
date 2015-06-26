
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xsyr2k routine. It is based on the TestABC class.
//
// =================================================================================================

#include "wrapper_clblas.h"
#include "correctness/testabc.h"

namespace clblast {
// =================================================================================================

// The correctness tester, containing the function calls to CLBlast and to clBLAS for comparison.
template <typename T>
void Xsyr2kTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [](const Arguments<T> &args,
                           const Buffer &a_mat, const Buffer &b_mat, const Buffer &c_mat,
                           CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    return Syr2k(args.layout, args.triangle, args.a_transpose,
                 args.n, args.k,
                 args.alpha,
                 a_mat(), args.a_offset, args.a_ld,
                 b_mat(), args.b_offset, args.b_ld,
                 args.beta,
                 c_mat(), args.c_offset, args.c_ld,
                 &queue_plain, &event);
  };

  // Creates the clBLAS lambda (for comparison)
  auto clblas_lambda = [](const Arguments<T> &args,
                          const Buffer &a_mat, const Buffer &b_mat, const Buffer &c_mat,
                          CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    auto status = clblasXsyr2k(static_cast<clblasOrder>(args.layout),
                               static_cast<clblasUplo>(args.triangle),
                               static_cast<clblasTranspose>(args.a_transpose),
                               args.n, args.k,
                               args.alpha,
                               a_mat(), args.a_offset, args.a_ld,
                               b_mat(), args.b_offset, args.b_ld,
                               args.beta,
                               c_mat(), args.c_offset, args.c_ld,
                               1, &queue_plain, 0, nullptr, &event);
    return static_cast<StatusCode>(status);
  };

  // Initializes the arguments relevant for this routine
  auto args = Arguments<T>{};
  const auto options = std::vector<std::string>{kArgN, kArgK, kArgLayout,
                                                kArgTriangle, kArgATransp,
                                                kArgALeadDim, kArgBLeadDim, kArgCLeadDim,
                                                kArgAOffset, kArgBOffset, kArgCOffset};

  // Creates a tester
  TestABC<T> tester{argc, argv, silent, name, options, clblast_lambda, clblas_lambda};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) {
    args.layout = layout;
    for (auto &triangle: {Triangle::kUpper, Triangle::kLower}) {
      args.triangle = triangle;
      for (auto &ab_transpose: {Transpose::kNo, Transpose::kYes}) { // No conjugate here since it is
        args.a_transpose = ab_transpose;                            // not supported by clBLAS
        args.b_transpose = ab_transpose;
        const auto case_name = ToString(layout)+" "+ToString(triangle)+" "+ToString(ab_transpose);

        // Runs the tests
        tester.TestRegular(args, case_name, true);
        tester.TestInvalidBufferSizes(args, case_name);
      }
    }
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::Xsyr2kTest<float>(argc, argv, false, "SSYR2K");
  clblast::Xsyr2kTest<double>(argc, argv, true, "DSYR2K");
  clblast::Xsyr2kTest<clblast::float2>(argc, argv, true, "CSYR2K");
  clblast::Xsyr2kTest<clblast::double2>(argc, argv, true, "ZSYR2K");
  return 0;
}

// =================================================================================================
