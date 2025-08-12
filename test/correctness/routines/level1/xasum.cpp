
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xasum.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXasum<float>, float, float>(argc, argv, false, "SASUM");
  errors += clblast::RunTests<clblast::TestXasum<double>, double, double>(argc, argv, true, "DASUM");
  errors += clblast::RunTests<clblast::TestXasum<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "ScASUM");
  errors += clblast::RunTests<clblast::TestXasum<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "DzASUM");
  errors +=
      clblast::RunTests<clblast::TestXasum<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HASUM");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
