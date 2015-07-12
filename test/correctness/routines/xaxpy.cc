
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xaxpy routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/xaxpy.h"

namespace clblast {
// =================================================================================================

// The correctness tester
template <typename T>
void RunTest(int argc, char *argv[], const bool silent, const std::string &name) {

  // Creates a tester
  TestBlas<T,T> tester{argc, argv, silent, name, TestXaxpy<T>::GetOptions(),
                       TestXaxpy<T>::RunRoutine, TestXaxpy<T>::RunReference,
                       TestXaxpy<T>::DownloadResult, TestXaxpy<T>::GetResultIndex,
                       TestXaxpy<T>::ResultID1, TestXaxpy<T>::ResultID2};

  // This variable holds the arguments relevant for this routine
  auto args = Arguments<T>{};

  // Creates the arguments vector for the regular tests
  auto regular_test_vector = std::vector<Arguments<T>>{};
  for (auto &n: tester.kVectorDims) { args.n = n;
    for (auto &x_inc: tester.kIncrements) { args.x_inc = x_inc;
      for (auto &x_offset: tester.kOffsets) { args.x_offset = x_offset;
        for (auto &y_inc: tester.kIncrements) { args.y_inc = y_inc;
          for (auto &y_offset: tester.kOffsets) { args.y_offset = y_offset;
            for (auto &alpha: tester.kAlphaValues) { args.alpha = alpha;
              args.x_size = TestXaxpy<T>::GetSizeX(args);
              args.y_size = TestXaxpy<T>::GetSizeY(args);
              if (args.x_size<1 || args.y_size<1) { continue; }
              regular_test_vector.push_back(args);
            }
          }
        }
      }
    }
  }

  // Creates the arguments vector for the invalid-buffer tests
  auto invalid_test_vector = std::vector<Arguments<T>>{};
  args.n = tester.kBufferSize;
  args.x_inc = args.y_inc = 1;
  args.x_offset = args.y_offset = 0;
  for (auto &x_size: tester.kVecSizes) { args.x_size = x_size;
    for (auto &y_size: tester.kVecSizes) { args.y_size = y_size;
      invalid_test_vector.push_back(args);
    }
  }

  // Runs the tests
  const auto case_name = "default";
  tester.TestRegular(regular_test_vector, case_name);
  tester.TestInvalid(invalid_test_vector, case_name);
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTest<float>(argc, argv, false, "SAXPY");
  clblast::RunTest<double>(argc, argv, true, "DAXPY");
  clblast::RunTest<clblast::float2>(argc, argv, true, "CAXPY");
  clblast::RunTest<clblast::double2>(argc, argv, true, "ZAXPY");
  return 0;
}

// =================================================================================================
