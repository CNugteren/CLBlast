
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xrotmg.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXrotmg<float>, float, float>(argc, argv, false, "SROTMG");
  errors += clblast::RunTests<clblast::TestXrotmg<double>, double, double>(argc, argv, true, "DROTMG");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
