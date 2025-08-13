
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/levelx/xaxpybatched.hpp"

#include "test/performance/client.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(command_line_args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      clblast::RunClient<clblast::TestXaxpyBatched<clblast::half>, clblast::half, clblast::half>(argc, argv);
      break;
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXaxpyBatched<float>, float, float>(argc, argv);
      break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXaxpyBatched<double>, double, double>(argc, argv);
      break;
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestXaxpyBatched<clblast::float2>, clblast::float2, clblast::float2>(argc, argv);
      break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestXaxpyBatched<clblast::double2>, clblast::double2, clblast::double2>(argc, argv);
      break;
  }
  return 0;
}

// =================================================================================================
