
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/levelx/xinvert.hpp"

#include "test/performance/client.hpp"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(command_line_args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      clblast::RunClient<clblast::TestXinvert<half>, half, half>(argc, argv);
      break;
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXinvert<float>, float, float>(argc, argv);
      break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXinvert<double>, double, double>(argc, argv);
      break;
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestXinvert<float2>, float2, float2>(argc, argv);
      break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestXinvert<double2>, double2, double2>(argc, argv);
      break;
  }
  return 0;
}

// =================================================================================================
