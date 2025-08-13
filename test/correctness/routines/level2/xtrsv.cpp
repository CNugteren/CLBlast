
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xtrsv.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXtrsv<float>, float, float>(argc, argv, false, "STRSV");
  errors += clblast::RunTests<clblast::TestXtrsv<double>, double, double>(argc, argv, true, "DTRSV");
  errors += clblast::RunTests<clblast::TestXtrsv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "CTRSV");
  errors += clblast::RunTests<clblast::TestXtrsv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZTRSV");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
