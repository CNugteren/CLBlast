
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level3/xher2k.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXher2k<clblast::float2,float>, clblast::float2, float>(argc, argv, false, "CHER2K");
  errors += clblast::RunTests<clblast::TestXher2k<clblast::double2,double>, clblast::double2, double>(argc, argv, true, "ZHER2K");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
