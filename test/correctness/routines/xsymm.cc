
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xsymm routine. It is based on the TestABC class.
//
// =================================================================================================

#include "wrapper_clblas.h"
#include "correctness/testabc.h"

namespace clblast {
// =================================================================================================

// The correctness tester, containing the function calls to CLBlast and to clBLAS for comparison.
template <typename T>
void XsymmTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates the CLBlast lambda
  auto clblast_lambda = [](const Arguments<T> &args,
                           const Buffer &a_mat, const Buffer &b_mat, const Buffer &c_mat,
                           CommandQueue &queue) -> StatusCode {
    auto queue_plain = queue();
    auto event = cl_event{};
    return Symm(args.layout, args.side, args.triangle,
                args.m, args.n,
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
    return static_cast<StatusCode>(status);
  };

  // Selects the platform and device on which to test (command-line options)
  auto help = std::string{"Options given/available:\n"};
  const auto platform_id = GetArgument(argc, argv, help, kArgPlatform, size_t{0});
  const auto device_id = GetArgument(argc, argv, help, kArgDevice, size_t{0});
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes the other arguments relevant for this routine
  auto args = Arguments<T>{};
  const auto options = std::vector<std::string>{kArgM, kArgN, kArgLayout,
                                                kArgSide, kArgTriangle,
                                                kArgALeadDim, kArgBLeadDim, kArgCLeadDim,
                                                kArgAOffset, kArgBOffset, kArgCOffset};

  // Creates a tester
  TestABC<T> tester{platform_id, device_id, name, options, clblast_lambda, clblas_lambda};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: {Layout::kRowMajor, Layout::kColMajor}) {
    args.layout = layout;
    for (auto &side: {Side::kLeft, Side::kRight}) {
      args.side = side;
      for (auto &triangle: {Triangle::kUpper, Triangle::kLower}) {
        args.triangle = triangle;
        const auto case_name = ToString(layout)+" "+ToString(side)+" "+ToString(triangle);

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
  clblast::XsymmTest<float>(argc, argv, false, "SSYMM");
  clblast::XsymmTest<double>(argc, argv, true, "DSYMM");
  //clblast::XsymmTest<float2>(argc, argv, true, "CSYMM");
  //clblast::XsymmTest<double2>(argc, argv, true, "ZSYMM");
  return 0;
}

// =================================================================================================
