
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains 'clinfo' like diagnostics specific for CLBlast (debugging)
//
// =================================================================================================

#include <cstdio>
#include <chrono>
#include <algorithm>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

template<typename F>
double TimeFunction(const size_t num_runs, F const &function) {
  auto timings = std::vector<double>(num_runs);
  for (auto &timing: timings) {
    const auto start_time = std::chrono::steady_clock::now();
    function();
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
  }
  return *std::min_element(timings.begin(), timings.end());

}

void OpenCLDiagnostics(int argc, char *argv[]) {
  auto arguments = RetrieveCommandLineArguments(argc, argv);

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  fprintf(stdout, "\n* %s\n", help.c_str());

  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Finds device information
  const auto device_type = GetDeviceType(device);
  const auto device_vendor = GetDeviceVendor(device);
  const auto device_architecture = GetDeviceArchitecture(device);
  const auto device_name = GetDeviceName(device);
  printf("\n --- OpenCL device naming:\n");
  printf("* Device type                   %s\n", device.Type().c_str());
  printf("* Device name                   %s\n", device.Name().c_str());
  printf("* Platform vendor               %s\n", platform.Vendor().c_str());
  printf("* Platform version              %s\n", platform.Version().c_str());

  // Prints the CLBlast specific device names
  printf("\n --- CLBlast device naming:\n");
  printf("* Device type                   %s\n", device_type.c_str());
  printf("* Device name                   %s\n", device_name.c_str());
  printf("* Device vendor                 %s\n", device_vendor.c_str());
  printf("* Device architecture           %s\n", device_architecture.c_str());

  // Simple OpenCL benchmarking
  constexpr auto kNumRuns = 20;
  printf("\n --- Some OpenCL library benchmarks (functions from clpp11.h):\n");
  printf("* queue.GetContext()            %.4lf ms\n", TimeFunction(kNumRuns, [&](){queue.GetContext();} ));
  printf("* queue.GetDevice()             %.4lf ms\n", TimeFunction(kNumRuns, [&](){queue.GetDevice();} ));
  printf("* device.Name()                 %.4lf ms\n", TimeFunction(kNumRuns, [&](){device.Name();} ));
  printf("* Buffer<float>(context, 1024)  %.4lf ms\n", TimeFunction(kNumRuns, [&](){Buffer<float>(context, 1024);} ));

  printf("\n");
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::OpenCLDiagnostics(argc, argv);
  return 0;
}

// =================================================================================================
