
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xspr2.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXspr2<float>, float, float>(argc, argv, false, "SSPR2");
  errors += clblast::RunTests<clblast::TestXspr2<double>, double, double>(argc, argv, true, "DSPR2");
  errors +=
      clblast::RunTests<clblast::TestXspr2<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSPR2");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
