
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xger OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/xger.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, 0, clblast::XgerGetTunerDefaults, clblast::XgerGetTunerSettings<half>, clblast::XgerTestValidArguments<half>, clblast::XgerSetConstraints, clblast::XgerComputeLocalMemSize<half>, clblast::XgerSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, 0, clblast::XgerGetTunerDefaults, clblast::XgerGetTunerSettings<float>, clblast::XgerTestValidArguments<float>, clblast::XgerSetConstraints, clblast::XgerComputeLocalMemSize<float>, clblast::XgerSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, 0, clblast::XgerGetTunerDefaults, clblast::XgerGetTunerSettings<double>, clblast::XgerTestValidArguments<double>, clblast::XgerSetConstraints, clblast::XgerComputeLocalMemSize<double>, clblast::XgerSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, 0, clblast::XgerGetTunerDefaults, clblast::XgerGetTunerSettings<float2>, clblast::XgerTestValidArguments<float2>, clblast::XgerSetConstraints, clblast::XgerComputeLocalMemSize<float2>, clblast::XgerSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, 0, clblast::XgerGetTunerDefaults, clblast::XgerGetTunerSettings<double2>, clblast::XgerTestValidArguments<double2>, clblast::XgerSetConstraints, clblast::XgerComputeLocalMemSize<double2>, clblast::XgerSetArguments<double2>); break;
  }
  return 0;
}

// =================================================================================================
