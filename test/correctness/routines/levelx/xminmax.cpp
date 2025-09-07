
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Ekansh Jain
//
// =================================================================================================

#include "test/routines/levelx/xminmax.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXminmax<float>, float, float>(argc, argv, false, "iSAMINMAX");
  errors += clblast::RunTests<clblast::TestXminmax<double>, double, double>(argc, argv, true, "iDMINMAX");
  errors += clblast::RunTests<clblast::TestXminmax<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                       "iCMINMAX");
  errors += clblast::RunTests<clblast::TestXminmax<clblast::double2>, clblast::double2, clblast::double2>(
      argc, argv, true, "iZMINMAX");
  errors += clblast::RunTests<clblast::TestXminmax<clblast::half>, clblast::half, clblast::half>(argc, argv, true,
                                                                                                 "iHMINMAX");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
