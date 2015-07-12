
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xhemm routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/level3/xhemm.h"

namespace clblast {
// =================================================================================================

// The correctness tester
template <typename T>
void RunTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates a tester
  TestBlas<T,T> tester{argc, argv, silent, name, TestXhemm<T>::GetOptions(),
                       TestXhemm<T>::RunRoutine, TestXhemm<T>::RunReference,
                       TestXhemm<T>::DownloadResult, TestXhemm<T>::GetResultIndex,
                       TestXhemm<T>::ResultID1, TestXhemm<T>::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<T>{};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) { args.layout = layout;
    for (auto &side: tester.kSides) { args.side = side;
      for (auto &triangle: tester.kTriangles) { args.triangle = triangle;

        // Creates the arguments vector for the regular tests
        auto regular_test_vector = std::vector<Arguments<T>>{};
        for (auto &m: tester.kMatrixDims) { args.m = m;
          for (auto &n: tester.kMatrixDims) { args.n = n;
            for (auto &a_ld: tester.kMatrixDims) { args.a_ld = a_ld;
              for (auto &a_offset: tester.kOffsets) { args.a_offset = a_offset;
                for (auto &b_ld: tester.kMatrixDims) { args.b_ld = b_ld;
                  for (auto &b_offset: tester.kOffsets) { args.b_offset = b_offset;
                    for (auto &c_ld: tester.kMatrixDims) { args.c_ld = c_ld;
                      for (auto &c_offset: tester.kOffsets) { args.c_offset = c_offset;
                        for (auto &alpha: tester.kAlphaValues) { args.alpha = alpha;
                          for (auto &beta: tester.kBetaValues) { args.beta = beta;
                            args.a_size = TestXhemm<T>::GetSizeA(args);
                            args.b_size = TestXhemm<T>::GetSizeB(args);
                            args.c_size = TestXhemm<T>::GetSizeC(args);
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
        args.m = args.n = tester.kBufferSize;
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
        const auto case_name = ToString(layout)+" "+ToString(side)+" "+ToString(triangle);
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
  clblast::RunTest<clblast::float2>(argc, argv, true, "CHEMM");
  clblast::RunTest<clblast::double2>(argc, argv, true, "ZHEMM");
  return 0;
}

// =================================================================================================
