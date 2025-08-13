
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the a simple pre-processor for the OpenCL kernels. This pre-processor is used
// in cases where the vendor's OpenCL compiler falls short in loop unrolling and array-to-register
// promotion. This pre-processor is specific for the CLBlast code making many assumptions.
//
// =================================================================================================

#ifndef CLBLAST_KERNEL_PREPROCESSOR_H_
#define CLBLAST_KERNEL_PREPROCESSOR_H_

#include <string>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

std::string PreprocessKernelSource(const std::string& kernel_source);

// =================================================================================================
}  // namespace clblast

// CLBLAST_KERNEL_PREPROCESSOR_H_
#endif
