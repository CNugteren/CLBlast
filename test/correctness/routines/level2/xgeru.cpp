
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xgeru.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXgeru<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, false,
                                                                                                     "CGERU");
  errors += clblast::RunTests<clblast::TestXgeru<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZGERU");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
