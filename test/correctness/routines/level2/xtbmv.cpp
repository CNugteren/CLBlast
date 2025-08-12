
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xtbmv.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtbmv<float>, float, float>(argc, argv, false, "STBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<double>, double, double>(argc, argv, true, "DTBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "CTBMV");
  errors += clblast::RunTests<clblast::TestXtbmv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZTBMV");
  errors +=
      clblast::RunTests<clblast::TestXtbmv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HTBMV");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
