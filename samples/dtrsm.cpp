
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the DTRSM routine. It is a stand-alone example, but it does
// require the Khronos C++ OpenCL API header file (downloaded by CMake). The example uses C++
// features, but CLBlast can also be used using the regular C-style OpenCL API.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
//
// =================================================================================================

#include <cstdio>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the C++ OpenCL API. If not yet available, it can be found here:
// https://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include "cl.hpp"

// Includes the CLBlast library
#include <clblast.h>

// =================================================================================================

// Example use of the double-precision Xtrsm routine DTRSM, solving A*X = alpha*B, storing the
// result in the memory of matrix B. Uses row-major storage (C-style).
int main() {

  // OpenCL platform/device settings
  const auto platform_id = 0;
  const auto device_id = 0;

  // Example TRSM arguments
  const size_t m = 4;
  const size_t n = 3;
  const double alpha = 1.0;
  const auto a_ld = m;
  const auto b_ld = n;

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
  auto event = cl_event{nullptr};

  // Populate host matrices with some example data
  auto host_a = std::vector<double>({1.0,  2.0,  1.0, -2.0,
                                    0.0, -1.0, -2.0,  0.0,
                                    0.0,  0.0,  1.0,  1.0,
                                    0.0,  0.0,  0.0, -1.0});
  auto host_b = std::vector<double>({-1.0, -1.0,  3.0,
                                     1.0, -3.0,  2.0,
                                     1.0,  1.0, -1.0,
                                     4.0, -1.0, -2.0});
  // Expected result:
  //   8 -5  2
  // -11  3  4
  //   5  0 -3
  //  -4  1  2

  // Copy the matrices to the device
  auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, host_a.size()*sizeof(double));
  auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, host_b.size()*sizeof(double));
  queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size()*sizeof(double), host_a.data());
  queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size()*sizeof(double), host_b.data());

  // Call the DTRSM routine. Note that the type of alpha and beta (double) determine the precision.
  auto queue_plain = queue();
  auto status = clblast::Trsm(clblast::Layout::kRowMajor, clblast::Side::kLeft,
                              clblast::Triangle::kUpper, clblast::Transpose::kNo,
                              clblast::Diagonal::kNonUnit,
                              m, n,
                              alpha,
                              device_a(), 0, a_ld,
                              device_b(), 0, b_ld,
                              &queue_plain, &event);

  // Retrieves the results
  if (status == clblast::StatusCode::kSuccess) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }
  queue.enqueueReadBuffer(device_b, CL_TRUE, 0, host_b.size()*sizeof(double), host_b.data());

  // Example completed. See "clblast.h" for status codes (0 -> success).
  printf("Completed TRSM with status %d and results:\n", static_cast<int>(status));
  for (auto i = size_t{0}; i < m; ++i) {
    for (auto j = size_t{0}; j < n; ++j) {
      printf("%3.0f ", host_b[i * b_ld + j]);
    }
    printf("\n");
  }
  return 0;
}

// =================================================================================================
