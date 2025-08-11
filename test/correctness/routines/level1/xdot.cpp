
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level1/xdot.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXdot<float>, float, float>(argc, argv, false, "SDOT");
  errors += clblast::RunTests<clblast::TestXdot<double>, double, double>(argc, argv, true, "DDOT");
  errors += clblast::RunTests<clblast::TestXdot<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HDOT");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
