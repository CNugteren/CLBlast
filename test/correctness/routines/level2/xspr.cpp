
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xspr.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXspr<float>, float, float>(argc, argv, false, "SSPR");
  errors += clblast::RunTests<clblast::TestXspr<double>, double, double>(argc, argv, true, "DSPR");
  errors += clblast::RunTests<clblast::TestXspr<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSPR");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
