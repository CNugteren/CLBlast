
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/levelx/xgemmstridedbatched.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXgemmStridedBatched<float>, float, float>(argc, argv, false,
                                                                                     "SGEMMSTRIDEDBATCHED");
  errors += clblast::RunTests<clblast::TestXgemmStridedBatched<double>, double, double>(argc, argv, true,
                                                                                        "DGEMMSTRIDEDBATCHED");
  errors += clblast::RunTests<clblast::TestXgemmStridedBatched<clblast::float2>, clblast::float2, clblast::float2>(
      argc, argv, true, "CGEMMSTRIDEDBATCHED");
  errors += clblast::RunTests<clblast::TestXgemmStridedBatched<clblast::double2>, clblast::double2, clblast::double2>(
      argc, argv, true, "ZGEMMSTRIDEDBATCHED");
  errors += clblast::RunTests<clblast::TestXgemmStridedBatched<clblast::half>, clblast::half, clblast::half>(
      argc, argv, true, "HGEMMSTRIDEDBATCHED");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
