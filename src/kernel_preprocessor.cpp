
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the OpenCL kernel preprocessor (see the header for more information).
//
// =================================================================================================

#include <vector>
#include <chrono>

#include "kernel_preprocessor.hpp"

namespace clblast {
// =================================================================================================

std::string PreprocessKernelSource(const std::string& kernel_source) {
  const auto processed_kernel = kernel_source;
  return processed_kernel;
}

// =================================================================================================
} // namespace clblast
