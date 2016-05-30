
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/level1/xamax.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTests<clblast::TestXamax<float>, float, float>(argc, argv, false, "iSAMAX");
  clblast::RunTests<clblast::TestXamax<double>, double, double>(argc, argv, true, "iDAMAX");
  clblast::RunTests<clblast::TestXamax<float2>, float2, float2>(argc, argv, true, "iCAMAX");
  clblast::RunTests<clblast::TestXamax<double2>, double2, double2>(argc, argv, true, "iZAMAX");
  clblast::RunTests<clblast::TestXamax<half>, half, half>(argc, argv, true, "iHAMAX");
  return 0;
}

// =================================================================================================
