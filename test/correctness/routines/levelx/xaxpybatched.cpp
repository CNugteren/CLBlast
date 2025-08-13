
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/levelx/xaxpybatched.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXaxpyBatched<float>, float, float>(argc, argv, false, "SAXPYBATCHED");
  errors += clblast::RunTests<clblast::TestXaxpyBatched<double>, double, double>(argc, argv, true, "DAXPYBATCHED");
  errors += clblast::RunTests<clblast::TestXaxpyBatched<clblast::float2>, clblast::float2, clblast::float2>(
      argc, argv, true, "CAXPYBATCHED");
  errors += clblast::RunTests<clblast::TestXaxpyBatched<clblast::double2>, clblast::double2, clblast::double2>(
      argc, argv, true, "ZAXPYBATCHED");
  errors += clblast::RunTests<clblast::TestXaxpyBatched<clblast::half>, clblast::half, clblast::half>(argc, argv, true,
                                                                                                      "HAXPYBATCHED");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
