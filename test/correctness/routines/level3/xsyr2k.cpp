
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level3/xsyr2k.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXsyr2k<float>, float, float>(argc, argv, false, "SSYR2K");
  errors += clblast::RunTests<clblast::TestXsyr2k<double>, double, double>(argc, argv, true, "DSYR2K");
  errors += clblast::RunTests<clblast::TestXsyr2k<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                      "CSYR2K");
  errors += clblast::RunTests<clblast::TestXsyr2k<clblast::double2>, clblast::double2, clblast::double2>(
      argc, argv, true, "ZSYR2K");
  errors +=
      clblast::RunTests<clblast::TestXsyr2k<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSYR2K");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
