
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
#include "routines/level3/xtrmm.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtrmm<float>, float, float>(argc, argv, false, "STRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<double>, double, double>(argc, argv, true, "DTRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<float2>, float2, float2>(argc, argv, true, "CTRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<double2>, double2, double2>(argc, argv, true, "ZTRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<half>, half, half>(argc, argv, true, "HTRMM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
