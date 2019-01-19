
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the convgemm kernels.
//
// =================================================================================================

#include "tuning/kernels/xconvgemm.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, V, clblast::XConvGemmGetTunerDefaults, clblast::XConvGemmGetTunerSettings<half>, clblast::XConvGemmTestValidArguments<half>, clblast::XConvGemmSetConstraints, clblast::XConvGemmComputeLocalMemSize<half>, clblast::XConvGemmSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, V, clblast::XConvGemmGetTunerDefaults, clblast::XConvGemmGetTunerSettings<float>, clblast::XConvGemmTestValidArguments<float>, clblast::XConvGemmSetConstraints, clblast::XConvGemmComputeLocalMemSize<float>, clblast::XConvGemmSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, V, clblast::XConvGemmGetTunerDefaults, clblast::XConvGemmGetTunerSettings<double>, clblast::XConvGemmTestValidArguments<double>, clblast::XConvGemmSetConstraints, clblast::XConvGemmComputeLocalMemSize<double>, clblast::XConvGemmSetArguments<double>); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  return 0;
}

// =================================================================================================
