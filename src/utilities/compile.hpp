
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the CLBlast way to compile a kernel from source, used for the library and for
// the auto-tuners.
//
// =================================================================================================

#ifndef CLBLAST_UTILITIES_COMPILE_H_
#define CLBLAST_UTILITIES_COMPILE_H_

#include <string>
#include <vector>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Compiles a program from source code
std::shared_ptr<Program> CompileFromSource(
                          const std::string &source_string, const Precision precision,
                          const std::string &routine_name,
                          const Device& device, const Context& context,
                          std::vector<std::string>& options,
                          const size_t run_preprocessor, // 0: platform dependent, 1: always, 2: never
                          const bool silent = false);

// =================================================================================================
} // namespace clblast

// CLBLAST_UTILITIES_COMPILE_H_
#endif
