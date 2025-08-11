
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/levelx/xim2col.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXim2col<float>, float, float>(argc, argv, false, "SIM2COL");
  errors += clblast::RunTests<clblast::TestXim2col<double>, double, double>(argc, argv, true, "DIM2COL");
  errors += clblast::RunTests<clblast::TestXim2col<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CIM2COL");
  errors += clblast::RunTests<clblast::TestXim2col<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZIM2COL");
  errors += clblast::RunTests<clblast::TestXim2col<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HIM2COL");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
