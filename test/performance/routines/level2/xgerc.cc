
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "performance/client.h"
#include "routines/level2/xgerc.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  switch(clblast::GetPrecision(argc, argv)) {
    case clblast::Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kSingle: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kDouble: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestXgerc<float2>, float2, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestXgerc<double2>, double2, double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
