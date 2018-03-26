
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the transpose OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/transpose_fast.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::TransposeGetTunerDefaults, clblast::TransposeGetTunerSettings<half>, clblast::TransposeTestValidArguments<half>, clblast::TransposeSetConstraints, clblast::TransposeComputeLocalMemSize<half>, clblast::TransposeSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::TransposeGetTunerDefaults, clblast::TransposeGetTunerSettings<float>, clblast::TransposeTestValidArguments<float>, clblast::TransposeSetConstraints, clblast::TransposeComputeLocalMemSize<float>, clblast::TransposeSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::TransposeGetTunerDefaults, clblast::TransposeGetTunerSettings<double>, clblast::TransposeTestValidArguments<double>, clblast::TransposeSetConstraints, clblast::TransposeComputeLocalMemSize<double>, clblast::TransposeSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::TransposeGetTunerDefaults, clblast::TransposeGetTunerSettings<float2>, clblast::TransposeTestValidArguments<float2>, clblast::TransposeSetConstraints, clblast::TransposeComputeLocalMemSize<float2>, clblast::TransposeSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::TransposeGetTunerDefaults, clblast::TransposeGetTunerSettings<double2>, clblast::TransposeTestValidArguments<double2>, clblast::TransposeSetConstraints, clblast::TransposeComputeLocalMemSize<double2>, clblast::TransposeSetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
