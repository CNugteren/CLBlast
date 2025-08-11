
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/correctness/testblas.hpp"
#include "test/routines/level3/xherk.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunTests<clblast::TestXherk<clblast::float2,float>, clblast::float2, float>(argc, argv, false, "CHERK");
  errors += clblast::RunTests<clblast::TestXherk<clblast::double2,double>, clblast::double2, double>(argc, argv, true, "ZHERK");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
