
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xsyr2k routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/xsyr2k.h"

namespace clblast {
// =================================================================================================

// The correctness tester
template <typename T>
void RunTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates a tester
  TestBlas<T> tester{argc, argv, silent, name, TestXsyr2k<T>::GetOptions(),
                     TestXsyr2k<T>::RunRoutine, TestXsyr2k<T>::RunReference,
                     TestXsyr2k<T>::DownloadResult, TestXsyr2k<T>::GetResultIndex,
                     TestXsyr2k<T>::ResultID1, TestXsyr2k<T>::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<T>{};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) { args.layout = layout;
    for (auto &triangle: tester.kTriangles) { args.triangle = triangle;
      for (auto &ab_transpose: {Transpose::kNo, Transpose::kYes}) { // No conjugate here since it
        args.a_transpose = ab_transpose;                            // is not supported by clBLAS
        args.b_transpose = ab_transpose;

        // Creates the arguments vector for the regular tests
        auto regular_test_vector = std::vector<Arguments<T>>{};
        for (auto &n: tester.kMatrixDims) { args.n = n;
          for (auto &k: tester.kMatrixDims) { args.k = k;
            for (auto &a_ld: tester.kMatrixDims) { args.a_ld = a_ld;
              for (auto &a_offset: tester.kOffsets) { args.a_offset = a_offset;
                for (auto &b_ld: tester.kMatrixDims) { args.b_ld = b_ld;
                  for (auto &b_offset: tester.kOffsets) { args.b_offset = b_offset;
                    for (auto &c_ld: tester.kMatrixDims) { args.c_ld = c_ld;
                      for (auto &c_offset: tester.kOffsets) { args.c_offset = c_offset;
                        for (auto &alpha: tester.kAlphaValues) { args.alpha = alpha;
                          for (auto &beta: tester.kBetaValues) { args.beta = beta;
                            args.a_size = TestXsyr2k<T>::GetSizeA(args);
                            args.b_size = TestXsyr2k<T>::GetSizeB(args);
                            args.c_size = TestXsyr2k<T>::GetSizeC(args);
                            if (args.a_size<1 || args.b_size<1 || args.c_size<1) { continue; }
                            regular_test_vector.push_back(args);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

        // Creates the arguments vector for the invalid-buffer tests
        auto invalid_test_vector = std::vector<Arguments<T>>{};
        args.n = args.k = tester.kBufferSize;
        args.a_ld = args.b_ld = args.c_ld = tester.kBufferSize;
        args.a_offset = args.b_offset = args.c_offset = 0;
        for (auto &a_size: tester.kMatSizes) { args.a_size = a_size;
          for (auto &b_size: tester.kMatSizes) { args.b_size = b_size;
            for (auto &c_size: tester.kMatSizes) { args.c_size = c_size;
              invalid_test_vector.push_back(args);
            }
          }
        }

        // Runs the tests
        const auto case_name = ToString(layout)+" "+ToString(triangle)+" "+ToString(ab_transpose);
        tester.TestRegular(regular_test_vector, case_name);
        tester.TestInvalid(invalid_test_vector, case_name);
      }
    }
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTest<float>(argc, argv, false, "SSYR2K");
  clblast::RunTest<double>(argc, argv, true, "DSYR2K");
  clblast::RunTest<clblast::float2>(argc, argv, true, "CSYR2K");
  clblast::RunTest<clblast::double2>(argc, argv, true, "ZSYR2K");
  return 0;
}

// =================================================================================================
