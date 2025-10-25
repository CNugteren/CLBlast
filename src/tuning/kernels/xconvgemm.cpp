
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the convgemm kernels.
//
// =================================================================================================

#include "tuning/kernels/xconvgemm.hpp"

#include "tuning/tuning.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(const int argc, char* argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch (clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf:
      clblast::Tuner<half>(argc, argv, V, clblast::XConvGemmGetTunerDefaults, clblast::XConvGemmGetTunerSettings<half>,
                           clblast::XConvGemmTestValidArguments<half>, clblast::XConvGemmSetConstraints,
                           clblast::XConvGemmComputeLocalMemSize<half>, clblast::XConvGemmSetArguments<half>);
      break;
    case clblast::Precision::kSingle:
      clblast::Tuner<float>(argc, argv, V, clblast::XConvGemmGetTunerDefaults,
                            clblast::XConvGemmGetTunerSettings<float>, clblast::XConvGemmTestValidArguments<float>,
                            clblast::XConvGemmSetConstraints, clblast::XConvGemmComputeLocalMemSize<float>,
                            clblast::XConvGemmSetArguments<float>);
      break;
    case clblast::Precision::kDouble:
      clblast::Tuner<double>(argc, argv, V, clblast::XConvGemmGetTunerDefaults,
                             clblast::XConvGemmGetTunerSettings<double>, clblast::XConvGemmTestValidArguments<double>,
                             clblast::XConvGemmSetConstraints, clblast::XConvGemmComputeLocalMemSize<double>,
                             clblast::XConvGemmSetArguments<double>);
      break;
    case clblast::Precision::kComplexSingle:
    case clblast::Precision::kComplexDouble:
    case clblast::Precision::kAny:
      break;
  }
}

// Main function (not within the clblast namespace)
int main(const int argc, char* argv[]) {
  try {
    StartVariation<1>(argc, argv);
    return 0;
  } catch (...) {
    return static_cast<int>(clblast::DispatchException());
  }
}

// =================================================================================================
