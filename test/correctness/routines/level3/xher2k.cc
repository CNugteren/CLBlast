
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xher2k routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/level3/xher2k.h"

// =================================================================================================

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTests<clblast::TestXher2k<float2,float>, float2, float>(argc, argv, false, "CHER2K");
  clblast::RunTests<clblast::TestXher2k<double2,double>, double2, double>(argc, argv, true, "ZHER2K");
  return 0;
}

// =================================================================================================
