
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xnrm2.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXnrm2<float>, float, float>(argc, argv, false, "SNRM2");
  errors += clblast::RunTests<clblast::TestXnrm2<double>, double, double>(argc, argv, true, "DNRM2");
  errors += clblast::RunTests<clblast::TestXnrm2<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true,
                                                                                                     "ScNRM2");
  errors += clblast::RunTests<clblast::TestXnrm2<clblast::double2>, clblast::double2, clblast::double2>(argc, argv,
                                                                                                        true, "DzNRM2");
  errors +=
      clblast::RunTests<clblast::TestXnrm2<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HNRM2");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
