
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/levelx/xcol2im.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXcol2im<float>, float, float>(argc, argv, false, "SCOL2IM");
  errors += clblast::RunTests<clblast::TestXcol2im<double>, double, double>(argc, argv, true, "DCOL2IM");
  errors += clblast::RunTests<clblast::TestXcol2im<clblast::float2>, clblast::float2, clblast::float2>(argc, argv, true, "CCOL2IM");
  errors += clblast::RunTests<clblast::TestXcol2im<clblast::double2>, clblast::double2, clblast::double2>(argc, argv, true, "ZCOL2IM");
  errors += clblast::RunTests<clblast::TestXcol2im<clblast::half>, clblast::half, clblast::half>(argc, argv, true, "HCOL2IM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
