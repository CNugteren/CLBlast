
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xsymv.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXsymv<float>, float, float>(argc, argv, false, "SSYMV");
  errors += clblast::RunTests<clblast::TestXsymv<double>, double, double>(argc, argv, true, "DSYMV");
  errors +=
      clblast::RunTests<clblast::TestXsymv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSYMV");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
