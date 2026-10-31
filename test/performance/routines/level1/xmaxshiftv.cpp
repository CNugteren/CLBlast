
// =================================================================================================
// This file is part of the CLBlast project.
// =================================================================================================

#include "test/routines/level1/xmaxshiftv.hpp"

#include "test/performance/client.hpp"

int main(int argc, char* argv[]) {
  const auto args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      clblast::RunClient<clblast::TestXmaxshiftv<clblast::half>, clblast::half, clblast::half>(argc, argv);
      break;
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestXmaxshiftv<float>, float, float>(argc, argv);
      break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestXmaxshiftv<double>, double, double>(argc, argv);
      break;
    default:
      return 1;
  }
  return 0;
}
