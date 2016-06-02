
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
#include "routines/level2/xger.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXger<float>, float, float>(argc, argv, false, "SGER");
  errors += clblast::RunTests<clblast::TestXger<double>, double, double>(argc, argv, true, "DGER");
  errors += clblast::RunTests<clblast::TestXger<half>, half, half>(argc, argv, true, "HGER");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
