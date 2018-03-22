
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xdot OpenCL kernels. Note that the results are
// not verified, since the result is not final and depends on the WGS2 parameter.
//
// =================================================================================================

#include "tuning/kernels/xdot.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, V, clblast::XdotGetTunerDefaults, clblast::XdotGetTunerSettings<half>, clblast::XdotTestValidArguments<half>, clblast::XdotSetConstraints, clblast::XdotComputeLocalMemSize<half>, clblast::XdotSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, V, clblast::XdotGetTunerDefaults, clblast::XdotGetTunerSettings<float>, clblast::XdotTestValidArguments<float>, clblast::XdotSetConstraints, clblast::XdotComputeLocalMemSize<float>, clblast::XdotSetArguments<float>); break;
    case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, V, clblast::XdotGetTunerDefaults, clblast::XdotGetTunerSettings<double>, clblast::XdotTestValidArguments<double>, clblast::XdotSetConstraints, clblast::XdotComputeLocalMemSize<double>, clblast::XdotSetArguments<double>); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, V, clblast::XdotGetTunerDefaults, clblast::XdotGetTunerSettings<float2>, clblast::XdotTestValidArguments<float2>, clblast::XdotSetConstraints, clblast::XdotComputeLocalMemSize<float2>, clblast::XdotSetArguments<float2>); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, V, clblast::XdotGetTunerDefaults, clblast::XdotGetTunerSettings<double2>, clblast::XdotTestValidArguments<double2>, clblast::XdotSetConstraints, clblast::XdotComputeLocalMemSize<double2>, clblast::XdotSetArguments<double2>); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  StartVariation<1>(argc, argv);
  StartVariation<2>(argc, argv);
  return 0;
}

// =================================================================================================
