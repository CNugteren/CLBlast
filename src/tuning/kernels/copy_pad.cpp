
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the pad OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/copy_pad.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::PadGetTunerDefaults, clblast::PadGetTunerSettings<half>, clblast::PadTestValidArguments<half>, clblast::PadSetConstraints, clblast::PadComputeLocalMemSize<half>, clblast::PadSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::PadGetTunerDefaults, clblast::PadGetTunerSettings<float>, clblast::PadTestValidArguments<float>, clblast::PadSetConstraints, clblast::PadComputeLocalMemSize<float>, clblast::PadSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::PadGetTunerDefaults, clblast::PadGetTunerSettings<double>, clblast::PadTestValidArguments<double>, clblast::PadSetConstraints, clblast::PadComputeLocalMemSize<double>, clblast::PadSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::PadGetTunerDefaults, clblast::PadGetTunerSettings<float2>, clblast::PadTestValidArguments<float2>, clblast::PadSetConstraints, clblast::PadComputeLocalMemSize<float2>, clblast::PadSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::PadGetTunerDefaults, clblast::PadGetTunerSettings<double2>, clblast::PadTestValidArguments<double2>, clblast::PadSetConstraints, clblast::PadComputeLocalMemSize<double2>, clblast::PadSetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
