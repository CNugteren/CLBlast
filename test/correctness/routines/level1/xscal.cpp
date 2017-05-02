
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
#include "test/routines/level1/xscal.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXscal<float>, float, float>(argc, argv, false, "SSCAL");
  errors += clblast::RunTests<clblast::TestXscal<double>, double, double>(argc, argv, true, "DSCAL");
  errors += clblast::RunTests<clblast::TestXscal<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CSCAL");
  errors += clblast::RunTests<clblast::TestXscal<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZSCAL");
  errors += clblast::RunTests<clblast::TestXscal<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSCAL");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
