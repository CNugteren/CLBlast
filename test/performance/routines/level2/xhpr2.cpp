
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/performance/client.hpp"
#include "test/routines/level2/xhpr2.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args, clblast::Precision::kComplexSingle)) {
    case clblast::Precision::kHalf: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kSingle: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kDouble: throw std::runtime_error("Unsupported precision mode");
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestXhpr2<clblast::float2>, clblast::float2, clblast::float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestXhpr2<clblast::double2>, clblast::double2, clblast::double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================
