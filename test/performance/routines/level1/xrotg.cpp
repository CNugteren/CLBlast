
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/routines/level1/xrotg.hpp"

#include "test/performance/client.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(command_line_args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXrotg<float>, float, float>(argc, argv);
      break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXrotg<double>, double, double>(argc, argv);
      break;
    case clblast::Precision::kComplexSingle:
      throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kComplexDouble:
      throw std::runtime_error("Unsupported precision mode");
  }
  return 0;
}

// =================================================================================================
