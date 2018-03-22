
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xgemv OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/xgemv.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, V, clblast::XgemvGetTunerDefaults, clblast::XgemvGetTunerSettings<half>, clblast::XgemvTestValidArguments<half>, clblast::XgemvSetConstraints, clblast::XgemvComputeLocalMemSize<half>, clblast::XgemvSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, V, clblast::XgemvGetTunerDefaults, clblast::XgemvGetTunerSettings<float>, clblast::XgemvTestValidArguments<float>, clblast::XgemvSetConstraints, clblast::XgemvComputeLocalMemSize<float>, clblast::XgemvSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, V, clblast::XgemvGetTunerDefaults, clblast::XgemvGetTunerSettings<double>, clblast::XgemvTestValidArguments<double>, clblast::XgemvSetConstraints, clblast::XgemvComputeLocalMemSize<double>, clblast::XgemvSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, V, clblast::XgemvGetTunerDefaults, clblast::XgemvGetTunerSettings<float2>, clblast::XgemvTestValidArguments<float2>, clblast::XgemvSetConstraints, clblast::XgemvComputeLocalMemSize<float2>, clblast::XgemvSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, V, clblast::XgemvGetTunerDefaults, clblast::XgemvGetTunerSettings<double2>, clblast::XgemvTestValidArguments<double2>, clblast::XgemvSetConstraints, clblast::XgemvComputeLocalMemSize<double2>, clblast::XgemvSetArguments<double2>); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  StartVariation<3>(argc, argv);
  return 0;
}

// =================================================================================================
