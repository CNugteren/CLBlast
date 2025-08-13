
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level3/xsyrk.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXsyrk<float>, float, float>(argc, argv, false, "SSYRK");
  errors += clblast::RunTests<clblast::TestXsyrk<double>, double, double>(argc, argv, true, "DSYRK");
  errors += clblast::RunTests<clblast::TestXsyrk<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "CSYRK");
  errors += clblast::RunTests<clblast::TestXsyrk<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZSYRK");
  errors +=
      clblast::RunTests<clblast::TestXsyrk<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSYRK");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
