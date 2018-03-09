
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the runtime tuning API. It is a stand-alone example, but it
// does require the Khronos C++ OpenCL API header file (downloaded by CMake).
//
// =================================================================================================

#include <cstdio>
#include <chrono>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the C++ OpenCL API. If not yet available, it can be found here:
// https://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include "cl.hpp"

// Includes the CLBlast library
#include <clblast.h>

// =================================================================================================

int main() {

  // OpenCL platform/device settings
  const auto platform_id = 0;
  const auto device_id = 0;

  // Example arguments
  const size_t m = 128;
  const size_t n = 64;
  const auto fraction = 1.0; // between 0.0 and 1.0

  // Initializes the OpenCL platform
  auto platforms = std::vector<cl::Platform>();
  cl::Platform::get(&platforms);
  if (platforms.size() == 0 || platform_id >= platforms.size()) { return 1; }
  auto platform = platforms[platform_id];

  // Initializes the OpenCL device
  auto devices = std::vector<cl::Device>();
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.size() == 0 || device_id >= devices.size()) { return 1; }
  auto device = devices[device_id];

  // Creates the OpenCL context, queue, and an event
  auto device_as_vector = std::vector<cl::Device>{device};
  auto context = cl::Context(device_as_vector);
  auto queue = cl::CommandQueue(context, device);

  // Performs the tuning
  printf("Starting the tuning...\n");
  std::unordered_map<std::string,size_t> parameters;
  auto queue_plain = queue();
  auto status = clblast::TuneCopy<float>(&queue_plain, m, n, fraction, parameters);

  // Tuning completed. See "clblast.h" for status codes (0 -> success).
  printf("Completed TuneCopy with status %d (0 == OK), found parameters:\n", static_cast<int>(status));
  for (const auto parameter: parameters) {
    printf(">  %s = %zu\n", parameter.first.c_str(), parameter.second);
  }

  // Set the new parameters
  status = clblast::OverrideParameters(device(), "Copy", clblast::Precision::kSingle, parameters);
  printf("Completed OverrideParameters with status %d (0 == OK)\n", static_cast<int>(status));
  return 0;
}

// =================================================================================================
