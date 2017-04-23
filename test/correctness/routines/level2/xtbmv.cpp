
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
#include "test/routines/level2/xtbmv.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtbmv<float>, float, float>(argc, argv, false, "STBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<double>, double, double>(argc, argv, true, "DTBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CTBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZTBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HTBMV");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
