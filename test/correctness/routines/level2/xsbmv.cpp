
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xsbmv.hpp"

#include "test/correctness/testblas.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXsbmv<float>, float, float>(argc, argv, false, "SSBMV");
  errors += clblast::RunTests<clblast::TestXsbmv<double>, double, double>(argc, argv, true, "DSBMV");
  errors +=
      clblast::RunTests<clblast::TestXsbmv<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HSBMV");
  if (errors > 0) {
    return 1;
  } else {
    return 0;
  }
}

// =================================================================================================
