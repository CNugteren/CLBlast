
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xamax.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXamax<float>, float, float>(argc, argv, false, "iSAMAX");
  errors += clblast::RunTests<clblast::TestXamax<double>, double, double>(argc, argv, true, "iDAMAX");
  errors += clblast::RunTests<clblast::TestXamax<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "iCAMAX");
  errors += clblast::RunTests<clblast::TestXamax<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "iZAMAX");
  errors +=
      clblast::RunTests<clblast::TestXamax<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "iHAMAX");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
