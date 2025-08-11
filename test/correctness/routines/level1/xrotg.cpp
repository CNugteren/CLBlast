
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level1/xrotg.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXrotg<float>, float, float>(argc, argv, false, "SROTG");
  errors += clblast::RunTests<clblast::TestXrotg<double>, double, double>(argc, argv, true, "DROTG");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
