
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level2/xsyr.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXsyr<float>, float, float>(argc, argv, false, "SSYR");
  errors += clblast::RunTests<clblast::TestXsyr<double>, double, double>(argc, argv, true, "DSYR");
  errors += clblast::RunTests<clblast::TestXsyr<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSYR");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
