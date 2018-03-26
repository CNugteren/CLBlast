
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
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::InvertGetTunerDefaults, clblast::InvertGetTunerSettings<half>, clblast::InvertTestValidArguments<half>, clblast::InvertSetConstraints, clblast::InvertComputeLocalMemSize<half>, clblast::InvertSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::InvertGetTunerDefaults, clblast::InvertGetTunerSettings<float>, clblast::InvertTestValidArguments<float>, clblast::InvertSetConstraints, clblast::InvertComputeLocalMemSize<float>, clblast::InvertSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::InvertGetTunerDefaults, clblast::InvertGetTunerSettings<double>, clblast::InvertTestValidArguments<double>, clblast::InvertSetConstraints, clblast::InvertComputeLocalMemSize<double>, clblast::InvertSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::InvertGetTunerDefaults, clblast::InvertGetTunerSettings<float2>, clblast::InvertTestValidArguments<float2>, clblast::InvertSetConstraints, clblast::InvertComputeLocalMemSize<float2>, clblast::InvertSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::InvertGetTunerDefaults, clblast::InvertGetTunerSettings<double2>, clblast::InvertTestValidArguments<double2>, clblast::InvertSetConstraints, clblast::InvertComputeLocalMemSize<double2>, clblast::InvertSetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
