
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
#include "test/routines/level3/xgemm.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXgemm<float>, float, float>(argc, argv, false, "SGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<double>, double, double>(argc, argv, true, "DGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<half>, half, half>(argc, argv, true, "HGEMM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
