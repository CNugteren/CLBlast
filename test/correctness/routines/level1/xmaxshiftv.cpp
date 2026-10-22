
// =================================================================================================
// This file is part of the CLBlast project.
// =================================================================================================

#include "test/routines/level1/xmaxshiftv.hpp"

#include "test/correctness/testblas.hpp"

int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXmaxshiftv<float>, float, float>(argc, argv, false, "SMAXSHIFTV");
  errors += clblast::RunTests<clblast::TestXmaxshiftv<double>, double, double>(argc, argv, true, "DMAXSHIFTV");
  errors += clblast::RunTests<clblast::TestXmaxshiftv<clblast::half>, clblast::half, clblast::half>(argc, argv, true,
                                                                                                    "HMAXSHIFTV");
  return errors > 0 ? 1 : 0;
}
