
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xgemv routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/xgemv.h"

namespace clblast {
// =================================================================================================

// The correctness tester
template <typename T>
void RunTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates a tester
  TestBlas<T,T> tester{argc, argv, silent, name, TestXgemv<T>::GetOptions(),
                       TestXgemv<T>::RunRoutine, TestXgemv<T>::RunReference,
                       TestXgemv<T>::DownloadResult, TestXgemv<T>::GetResultIndex,
                       TestXgemv<T>::ResultID1, TestXgemv<T>::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<T>{};

  // Loops over the test-cases from a data-layout point of view
  for (auto &layout: tester.kLayouts) { args.layout = layout;
    for (auto &a_transpose: tester.kTransposes) { args.a_transpose = a_transpose;

      // Creates the arguments vector for the regular tests
      auto regular_test_vector = std::vector<Arguments<T>>{};
      for (auto &m: tester.kMatrixVectorDims) { args.m = m;
        for (auto &n: tester.kMatrixVectorDims) { args.n = n;
          for (auto &a_ld: tester.kMatrixVectorDims) { args.a_ld = a_ld;
            for (auto &a_offset: tester.kOffsets) { args.a_offset = a_offset;
              for (auto &x_inc: tester.kIncrements) { args.x_inc = x_inc;
                for (auto &x_offset: tester.kOffsets) { args.x_offset = x_offset;
                  for (auto &y_inc: tester.kIncrements) { args.y_inc = y_inc;
                    for (auto &y_offset: tester.kOffsets) { args.y_offset = y_offset;
                      for (auto &alpha: tester.kAlphaValues) { args.alpha = alpha;
                        for (auto &beta: tester.kBetaValues) { args.beta = beta;
                          args.a_size = TestXgemv<T>::GetSizeA(args);
                          args.x_size = TestXgemv<T>::GetSizeX(args);
                          args.y_size = TestXgemv<T>::GetSizeY(args);
                          if (args.a_size<1 || args.x_size<1 || args.y_size<1) { continue; }
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
      args.a_ld = tester.kBufferSize;
      args.x_inc = args.y_inc = 1;
      args.a_offset = args.x_offset = args.y_offset = 0;
      for (auto &a_size: tester.kMatSizes) { args.a_size = a_size;
        for (auto &x_size: tester.kVecSizes) { args.x_size = x_size;
          for (auto &y_size: tester.kVecSizes) { args.y_size = y_size;
            invalid_test_vector.push_back(args);
          }
        }
      }

      // Runs the tests
      const auto case_name = ToString(layout)+" "+ToString(a_transpose);
      tester.TestRegular(regular_test_vector, case_name);
      tester.TestInvalid(invalid_test_vector, case_name);
    }
  }
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTest<float>(argc, argv, false, "SGEMV");
  clblast::RunTest<double>(argc, argv, true, "DGEMV");
  clblast::RunTest<clblast::float2>(argc, argv, true, "CGEMV");
  clblast::RunTest<clblast::double2>(argc, argv, true, "ZGEMV");
  return 0;
}

// =================================================================================================
