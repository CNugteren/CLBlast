
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
#include "test/routines/levelx/xinvert.hpp"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXinvert<float>, float, float>(argc, argv, false, "SINVERT");
  errors += clblast::RunTests<clblast::TestXinvert<double>, double, double>(argc, argv, true, "DINVERT");
  errors += clblast::RunTests<clblast::TestXinvert<float2>, float2, float2>(argc, argv, true, "CINVERT");
  errors += clblast::RunTests<clblast::TestXinvert<double2>, double2, double2>(argc, argv, true, "ZINVERT");
  errors += clblast::RunTests<clblast::TestXinvert<half>, half, half>(argc, argv, true, "HINVERT");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
