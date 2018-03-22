
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the direct xgemm kernels.
//
// =================================================================================================

#include "tuning/kernels/xgemm_direct.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, V, clblast::XgemmDirectGetTunerDefaults, clblast::XgemmDirectGetTunerSettings<half>, clblast::XgemmDirectTestValidArguments<half>, clblast::XgemmDirectSetConstraints, clblast::XgemmDirectComputeLocalMemSize<half>, clblast::XgemmDirectSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, V, clblast::XgemmDirectGetTunerDefaults, clblast::XgemmDirectGetTunerSettings<float>, clblast::XgemmDirectTestValidArguments<float>, clblast::XgemmDirectSetConstraints, clblast::XgemmDirectComputeLocalMemSize<float>, clblast::XgemmDirectSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, V, clblast::XgemmDirectGetTunerDefaults, clblast::XgemmDirectGetTunerSettings<double>, clblast::XgemmDirectTestValidArguments<double>, clblast::XgemmDirectSetConstraints, clblast::XgemmDirectComputeLocalMemSize<double>, clblast::XgemmDirectSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, V, clblast::XgemmDirectGetTunerDefaults, clblast::XgemmDirectGetTunerSettings<float2>, clblast::XgemmDirectTestValidArguments<float2>, clblast::XgemmDirectSetConstraints, clblast::XgemmDirectComputeLocalMemSize<float2>, clblast::XgemmDirectSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, V, clblast::XgemmDirectGetTunerDefaults, clblast::XgemmDirectGetTunerSettings<double2>, clblast::XgemmDirectTestValidArguments<double2>, clblast::XgemmDirectSetConstraints, clblast::XgemmDirectComputeLocalMemSize<double2>, clblast::XgemmDirectSetArguments<double2>); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  return 0;
}

// =================================================================================================
