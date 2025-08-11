
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level1/xdotu.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXdotu<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, false, "CDOTU");
  errors += clblast::RunTests<clblast::TestXdotu<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZDOTU");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
