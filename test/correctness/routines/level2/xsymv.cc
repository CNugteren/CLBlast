
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the Xsymv routine.
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/level2/xsymv.h"

// =================================================================================================

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTests<clblast::TestXsymv<float>, float, float>(argc, argv, false, "SSYMV");
  clblast::RunTests<clblast::TestXsymv<double>, double, double>(argc, argv, true, "DSYMV");
  return 0;
}

// =================================================================================================
