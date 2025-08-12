
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xcopy.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXcopy<float>, float, float>(argc, argv, false, "SCOPY");
  errors += clblast::RunTests<clblast::TestXcopy<double>, double, double>(argc, argv, true, "DCOPY");
  errors += clblast::RunTests<clblast::TestXcopy<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "CCOPY");
  errors += clblast::RunTests<clblast::TestXcopy<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZCOPY");
  errors +=
      clblast::RunTests<clblast::TestXcopy<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HCOPY");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
