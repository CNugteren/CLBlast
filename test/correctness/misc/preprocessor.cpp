
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the tests for the simple integrated OpenCL pre-processor
//
// =================================================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>

#include "utilities/utilities.hpp"
#include "utilities/compile.hpp"
#include "kernel_preprocessor.hpp"

namespace clblast {
// =================================================================================================

size_t RunPreprocessor(int argc, char *argv[], const bool silent,
                       const std::string &kernel_name, const std::string &kernel_source,
                       const Precision precision) {
  auto errors = size_t{0};
  auto passed = size_t{0};
  fprintf(stdout, "* Testing simple OpenCL pre-processor for '%s'\n", kernel_name.c_str());

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  auto arguments = RetrieveCommandLineArguments(argc, argv);
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  const auto context = Context(device);

  // Verifies that the current kernel compiles properly (assumes so, otherwise throws an error)
  auto compiler_options_ref = std::vector<std::string>();
  const auto program_ref = CompileFromSource(kernel_source, precision, kernel_name,
                                             device, context, compiler_options_ref);

  // Runs the pre-processor
  const auto processed_source = PreprocessKernelSource(kernel_source);

  // Verifies that the new kernel compiles properly
  try {
    auto compiler_options = std::vector<std::string>();
    const auto program = CompileFromSource(processed_source, precision, kernel_name,
                                           device, context, compiler_options);
    passed++;
  } catch (...) {
    fprintf(stdout, "* ERROR: Compilation warnings/errors with pre-processed kernel\n");
    errors++;
  }

  // Prints and returns the statistics
  std::cout << "    " << passed << " test(s) passed" << std::endl;
  std::cout << "    " << errors << " test(s) failed" << std::endl;
  std::cout << std::endl;
  return errors;
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  const auto xaxpy_sources =
  #include "../src/kernels/level1/level1.opencl"
  #include "../src/kernels/level1/xaxpy.opencl"
      ;
  errors += clblast::RunPreprocessor(argc, argv, false,
                                     "XaxpyFastest", xaxpy_sources, clblast::Precision::kSingle);
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
