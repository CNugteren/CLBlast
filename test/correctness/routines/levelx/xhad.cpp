
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/levelx/xhad.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXhad<float>, float, float>(argc, argv, false, "SHAD");
  errors += clblast::RunTests<clblast::TestXhad<double>, double, double>(argc, argv, true, "DHAD");
  errors +=
      clblast::RunTests<clblast::TestXhad<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CHAD");
  errors += clblast::RunTests<clblast::TestXhad<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true,
                                                                                                       "ZHAD");
  errors += clblast::RunTests<clblast::TestXhad<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HHAD");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
