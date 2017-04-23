
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
#include "test/routines/level2/xtrmv.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtrmv<float>, float, float>(argc, argv, false, "STRMV");
  errors += clblast::RunTests<clblast::TestXtrmv<double>, double, double>(argc, argv, true, "DTRMV");
  errors += clblast::RunTests<clblast::TestXtrmv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CTRMV");
  errors += clblast::RunTests<clblast::TestXtrmv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZTRMV");
  errors += clblast::RunTests<clblast::TestXtrmv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HTRMV");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
