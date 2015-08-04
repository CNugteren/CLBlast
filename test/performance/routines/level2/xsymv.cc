
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsymv command-line interface performance tester.
//
// =================================================================================================

#include "performance/client.h"
#include "routines/level2/xsymv.h"

// =================================================================================================

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  switch(clblast::GetPrecision(argc, argv)) {
    case clblast::Precision::kHalf:
      throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXsymv<float>, float, float>(argc, argv); break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXsymv<double>, double, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle:
      throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kComplexDouble:
      throw std::runtime_error("Unsupported precision mode");
  }
  return 0;
}

// =================================================================================================
