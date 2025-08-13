
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xgerc.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXgerc<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, false,
                                                                                                     "CGERC");
  errors += clblast::RunTests<clblast::TestXgerc<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZGERC");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
