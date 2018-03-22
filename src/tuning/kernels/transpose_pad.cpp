
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the pad-transpose OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/transpose_pad.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::PadtransposeGetTunerDefaults, clblast::PadtransposeGetTunerSettings<half>, clblast::PadtransposeTestValidArguments<half>, clblast::PadtransposeSetConstraints, clblast::PadtransposeComputeLocalMemSize<half>, clblast::PadtransposeSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::PadtransposeGetTunerDefaults, clblast::PadtransposeGetTunerSettings<float>, clblast::PadtransposeTestValidArguments<float>, clblast::PadtransposeSetConstraints, clblast::PadtransposeComputeLocalMemSize<float>, clblast::PadtransposeSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::PadtransposeGetTunerDefaults, clblast::PadtransposeGetTunerSettings<double>, clblast::PadtransposeTestValidArguments<double>, clblast::PadtransposeSetConstraints, clblast::PadtransposeComputeLocalMemSize<double>, clblast::PadtransposeSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::PadtransposeGetTunerDefaults, clblast::PadtransposeGetTunerSettings<float2>, clblast::PadtransposeTestValidArguments<float2>, clblast::PadtransposeSetConstraints, clblast::PadtransposeComputeLocalMemSize<float2>, clblast::PadtransposeSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::PadtransposeGetTunerDefaults, clblast::PadtransposeGetTunerSettings<double2>, clblast::PadtransposeTestValidArguments<double2>, clblast::PadtransposeSetConstraints, clblast::PadtransposeComputeLocalMemSize<double2>, clblast::PadtransposeSetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
