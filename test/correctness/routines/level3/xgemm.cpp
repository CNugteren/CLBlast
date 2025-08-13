
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level3/xgemm.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXgemm<1, float>, float, float>(argc, argv, false, "SGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<1, double>, double, double>(argc, argv, true, "DGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<1, clblast::float2>, clblast::float2, clblast::float2>(argc, argv,
                                                                                                        true, "CGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<1, clblast::double2>, clblast::double2, clblast::double2>(
      argc, argv, true, "ZGEMM");
  errors +=
      clblast::RunTests<clblast::TestXgemm<1, clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<2, float>, float, float>(argc, argv, true, "SGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<2, double>, double, double>(argc, argv, true, "DGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<2, clblast::float2>, clblast::float2, clblast::float2>(argc, argv,
                                                                                                        true, "CGEMM");
  errors += clblast::RunTests<clblast::TestXgemm<2, clblast::double2>, clblast::double2, clblast::double2>(
      argc, argv, true, "ZGEMM");
  errors +=
      clblast::RunTests<clblast::TestXgemm<2, clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HGEMM");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
