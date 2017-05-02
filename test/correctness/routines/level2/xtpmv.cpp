
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
#include "test/routines/level2/xtpmv.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtpmv<float>, float, float>(argc, argv, false, "STPMV");
  errors += clblast::RunTests<clblast::TestXtpmv<double>, double, double>(argc, argv, true, "DTPMV");
  errors += clblast::RunTests<clblast::TestXtpmv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CTPMV");
  errors += clblast::RunTests<clblast::TestXtpmv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZTPMV");
  errors += clblast::RunTests<clblast::TestXtpmv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HTPMV");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
