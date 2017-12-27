
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains a simple test to compile the invert kernel.
//
// =================================================================================================

#include <string>
#include <vector>
#include <cstdio>

#include "utilities/utilities.hpp"
#include "routines/levelx/xinvert.hpp"

namespace clblast {
// =================================================================================================

template <typename T>
size_t CompileInvertKernels(int argc, char *argv[], const bool silent) {

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  auto arguments = RetrieveCommandLineArguments(argc, argv);
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));

  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }

  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Compiles the invert kernels
  auto diagonal_invert_event = Event();
  auto inverter = Xinvert<T>(queue, diagonal_invert_event.pointer());

  // Report and return
  printf("\n");
  printf("    1 test(s) passed\n");
  printf("    0 test(s) failed\n");
  printf("\n");
  return 0;
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::CompileInvertKernels<float>(argc, argv, false);
  errors += clblast::CompileInvertKernels<clblast::float2>(argc, argv, true);
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================
