
// =================================================================================================
// This file is part of the CLBlast project.
// =================================================================================================

#include "test/routines/level3/xmaxshiftm.hpp"

#include "test/correctness/testblas.hpp"

int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXmaxshiftm<float>, float, float>(argc, argv, false, "SMAXSHIFTM");
  errors += clblast::RunTests<clblast::TestXmaxshiftm<double>, double, double>(argc, argv, true, "DMAXSHIFTM");
  errors += clblast::RunTests<clblast::TestXmaxshiftm<clblast::half>, clblast::half, clblast::half>(argc, argv, true,
                                                                                                    "HMAXSHIFTM");
  return errors > 0 ? 1 : 0;
}
