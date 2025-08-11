
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level1/xrot.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXrot<float>, float, float>(argc, argv, false, "SROT");
  errors += clblast::RunTests<clblast::TestXrot<double>, double, double>(argc, argv, true, "DROT");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
