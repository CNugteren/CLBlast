
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xherk routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/level3/xherk.h"

namespace clblast {
// =================================================================================================

// The correctness tester
template <typename T, typename U>
void RunTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates a tester
  TestBlas<T,U> tester{argc, argv, silent, name, TestXherk<T,U>::GetOptions(),
                       TestXherk<T,U>::RunRoutine, TestXherk<T,U>::RunReference,
                       TestXherk<T,U>::DownloadResult, TestXherk<T,U>::GetResultIndex,
                       TestXherk<T,U>::ResultID1, TestXherk<T,U>::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<U>{};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) { args.layout = layout;
    for (auto &triangle: tester.kTriangles) { args.triangle = triangle;
      for (auto &a_transpose: {Transpose::kNo, Transpose::kConjugate}) { // Regular transpose not a
        args.a_transpose = a_transpose;                                  // valid BLAS option

        // Creates the arguments vector for the regular tests
        auto regular_test_vector = std::vector<Arguments<U>>{};
        for (auto &n: tester.kMatrixDims) { args.n = n;
          for (auto &k: tester.kMatrixDims) { args.k = k;
            for (auto &a_ld: tester.kMatrixDims) { args.a_ld = a_ld;
              for (auto &a_offset: tester.kOffsets) { args.a_offset = a_offset;
                for (auto &c_ld: tester.kMatrixDims) { args.c_ld = c_ld;
                  for (auto &c_offset: tester.kOffsets) { args.c_offset = c_offset;
                    for (auto &alpha: tester.kAlphaValues) { args.alpha = alpha;
                      for (auto &beta: tester.kBetaValues) { args.beta = beta;
                        args.a_size = TestXherk<T,U>::GetSizeA(args);
                        args.c_size = TestXherk<T,U>::GetSizeC(args);
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
        auto invalid_test_vector = std::vector<Arguments<U>>{};
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
  clblast::RunTest<clblast::float2,float>(argc, argv, false, "CHERK");
  clblast::RunTest<clblast::double2,double>(argc, argv, true, "ZHERK");
  return 0;
}

// =================================================================================================
