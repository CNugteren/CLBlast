
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level3/xtrmm.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtrmm<float>, float, float>(argc, argv, false, "STRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<double>, double, double>(argc, argv, true, "DTRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "CTRMM");
  errors += clblast::RunTests<clblast::TestXtrmm<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZTRMM");
  errors +=
      clblast::RunTests<clblast::TestXtrmm<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HTRMM");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
