
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under the MIT license. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xsyrk routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/xsyrk.h"

namespace clblast {
// =================================================================================================

// The correctness tester
template <typename T>
void RunTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates a tester
  TestBlas<T> tester{argc, argv, silent, name, TestXsyrk<T>::GetOptions(),
                     TestXsyrk<T>::RunRoutine, TestXsyrk<T>::RunReference,
                     TestXsyrk<T>::DownloadResult, TestXsyrk<T>::GetResultIndex,
                     TestXsyrk<T>::ResultID1, TestXsyrk<T>::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<T>{};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) { args.layout = layout;
    for (auto &triangle: tester.kTriangles) { args.triangle = triangle;
      for (auto &a_transpose: {Transpose::kNo, Transpose::kYes}) { // No conjugate here since it
        args.a_transpose = a_transpose;                            // is not supported by clBLAS

        // Creates the arguments vector for the regular tests
        auto regular_test_vector = std::vector<Arguments<T>>{};
        for (auto &n: tester.kMatrixDims) { args.n = n;
          for (auto &k: tester.kMatrixDims) { args.k = k;
            for (auto &a_ld: tester.kMatrixDims) { args.a_ld = a_ld;
              for (auto &a_offset: tester.kOffsets) { args.a_offset = a_offset;
                for (auto &c_ld: tester.kMatrixDims) { args.c_ld = c_ld;
                  for (auto &c_offset: tester.kOffsets) { args.c_offset = c_offset;
                    for (auto &alpha: tester.kAlphaValues) { args.alpha = alpha;
                      for (auto &beta: tester.kBetaValues) { args.beta = beta;
                        args.a_size = TestXsyrk<T>::GetSizeA(args);
                        args.c_size = TestXsyrk<T>::GetSizeC(args);
                        if (args.a_size<1 || args.c_size<1) { continue; }
                        regular_test_vector.push_back(args);
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
        args.a_ld = args.c_ld = tester.kBufferSize;
        args.a_offset = args.c_offset = 0;
        for (auto &a_size: tester.kMatSizes) { args.a_size = a_size;
          for (auto &c_size: tester.kMatSizes) { args.c_size = c_size;
            invalid_test_vector.push_back(args);
          }
        }

        // Runs the tests
        const auto case_name = ToString(layout)+" "+ToString(triangle)+" "+ToString(a_transpose);
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
  clblast::RunTest<float>(argc, argv, false, "SSYRK");
  clblast::RunTest<double>(argc, argv, true, "DSYRK");
  clblast::RunTest<clblast::float2>(argc, argv, true, "CSYRK");
  clblast::RunTest<clblast::double2>(argc, argv, true, "ZSYRK");
  return 0;
}

// =================================================================================================
