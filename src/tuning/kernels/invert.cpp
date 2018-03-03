
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the invert OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/invert.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<half>, clblast::TestValidArguments<half>, clblast::SetConstraints, clblast::SetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<float>, clblast::TestValidArguments<float>, clblast::SetConstraints, clblast::SetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<double>, clblast::TestValidArguments<double>, clblast::SetConstraints, clblast::SetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<float2>, clblast::TestValidArguments<float2>, clblast::SetConstraints, clblast::SetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::GetTunerDefaults, clblast::GetTunerSettings<double2>, clblast::TestValidArguments<double2>, clblast::SetConstraints, clblast::SetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
