
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xtrmv.hpp"

#include "test/performance/client.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(command_line_args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      clblast::RunClient<clblast::TestXtrmv<clblast::half>, clblast::half, clblast::half>(argc, argv);
      break;
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXtrmv<float>, float, float>(argc, argv);
      break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXtrmv<double>, double, double>(argc, argv);
      break;
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestXtrmv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv);
      break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestXtrmv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv);
      break;
  }
  return 0;
}

// =================================================================================================
