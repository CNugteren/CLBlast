
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level2/xhbmv.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXhbmv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, false, "CHBMV");
  errors += clblast::RunTests<clblast::TestXhbmv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZHBMV");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
