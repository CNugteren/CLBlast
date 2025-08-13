
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xdotc.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXdotc<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, false,
                                                                                                     "CDOTC");
  errors += clblast::RunTests<clblast::TestXdotc<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZDOTC");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
