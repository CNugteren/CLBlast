
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xswap.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXswap<float>, float, float>(argc, argv, false, "SSWAP");
  errors += clblast::RunTests<clblast::TestXswap<double>, double, double>(argc, argv, true, "DSWAP");
  errors += clblast::RunTests<clblast::TestXswap<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "CSWAP");
  errors += clblast::RunTests<clblast::TestXswap<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "ZSWAP");
  errors +=
      clblast::RunTests<clblast::TestXswap<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSWAP");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
