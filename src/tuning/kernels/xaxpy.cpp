
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xaxpy OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/xaxpy.hpp"


#include "tuning/tuning.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char* argv[]) {
  try {
    const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
    switch (clblast::GetPrecision(command_line_args)) {
      case clblast::Precision::kHalf:
        clblast::Tuner<half>(argc, argv, 0, clblast::XaxpyGetTunerDefaults, clblast::XaxpyGetTunerSettings<half>,
                             clblast::XaxpyTestValidArguments<half>, clblast::XaxpySetConstraints,
                             clblast::XaxpyComputeLocalMemSize<half>, clblast::XaxpySetArguments<half>);
        break;
      case clblast::Precision::kSingle:
        clblast::Tuner<float>(argc, argv, 0, clblast::XaxpyGetTunerDefaults, clblast::XaxpyGetTunerSettings<float>,
                              clblast::XaxpyTestValidArguments<float>, clblast::XaxpySetConstraints,
                              clblast::XaxpyComputeLocalMemSize<float>, clblast::XaxpySetArguments<float>);
        break;
      case clblast::Precision::kDouble:
        clblast::Tuner<double>(argc, argv, 0, clblast::XaxpyGetTunerDefaults, clblast::XaxpyGetTunerSettings<double>,
                               clblast::XaxpyTestValidArguments<double>, clblast::XaxpySetConstraints,
                               clblast::XaxpyComputeLocalMemSize<double>, clblast::XaxpySetArguments<double>);
        break;
      case clblast::Precision::kComplexSingle:
        clblast::Tuner<float2>(argc, argv, 0, clblast::XaxpyGetTunerDefaults, clblast::XaxpyGetTunerSettings<float2>,
                               clblast::XaxpyTestValidArguments<float2>, clblast::XaxpySetConstraints,
                               clblast::XaxpyComputeLocalMemSize<float2>, clblast::XaxpySetArguments<float2>);
        break;
      case clblast::Precision::kComplexDouble:
        clblast::Tuner<double2>(argc, argv, 0, clblast::XaxpyGetTunerDefaults, clblast::XaxpyGetTunerSettings<double2>,
                                clblast::XaxpyTestValidArguments<double2>, clblast::XaxpySetConstraints,
                                clblast::XaxpyComputeLocalMemSize<double2>, clblast::XaxpySetArguments<double2>);
        break;
    }
    return 0;
  } catch (...) {
    return static_cast<int>(clblast::DispatchException());
  }
}

// =================================================================================================
