
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the copy OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/copy_fast.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::CopyGetTunerDefaults, clblast::CopyGetTunerSettings<half>, clblast::CopyTestValidArguments<half>, clblast::CopySetConstraints, clblast::CopyComputeLocalMemSize<half>, clblast::CopySetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::CopyGetTunerDefaults, clblast::CopyGetTunerSettings<float>, clblast::CopyTestValidArguments<float>, clblast::CopySetConstraints, clblast::CopyComputeLocalMemSize<float>, clblast::CopySetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::CopyGetTunerDefaults, clblast::CopyGetTunerSettings<double>, clblast::CopyTestValidArguments<double>, clblast::CopySetConstraints, clblast::CopyComputeLocalMemSize<double>, clblast::CopySetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::CopyGetTunerDefaults, clblast::CopyGetTunerSettings<float2>, clblast::CopyTestValidArguments<float2>, clblast::CopySetConstraints, clblast::CopyComputeLocalMemSize<float2>, clblast::CopySetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::CopyGetTunerDefaults, clblast::CopyGetTunerSettings<double2>, clblast::CopyTestValidArguments<double2>, clblast::CopySetConstraints, clblast::CopyComputeLocalMemSize<double2>, clblast::CopySetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
