
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level2/xtpsv.hpp"

#include "test/performance/client.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(command_line_args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXtpsv<float>, float, float>(argc, argv);
      break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXtpsv<double>, double, double>(argc, argv);
      break;
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestXtpsv<clblast::float2>, clblast::float2, clblast::float2>(argc, argv);
      break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestXtpsv<clblast::double2>, clblast::double2, clblast::double2>(argc, argv);
      break;
  }
  return 0;
}

// =================================================================================================
