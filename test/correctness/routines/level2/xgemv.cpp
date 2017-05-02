
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
#include "test/routines/level2/xgemv.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXgemv<float>, float, float>(argc, argv, false, "SGEMV");
  errors += clblast::RunTests<clblast::TestXgemv<double>, double, double>(argc, argv, true, "DGEMV");
  errors += clblast::RunTests<clblast::TestXgemv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CGEMV");
  errors += clblast::RunTests<clblast::TestXgemv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZGEMV");
  errors += clblast::RunTests<clblast::TestXgemv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HGEMV");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
