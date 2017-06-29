
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the batched SGEMM routine. It is a stand-alone example, but it
// does require the Khronos C++ OpenCL API header file (downloaded by CMake). The example uses C++
// features, but CLBlast can also be used using the regular C-style OpenCL API.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
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

// Example use of the single-precision batched SGEMM routine
int main() {

  // OpenCL platform/device settings
  const auto platform_id = 0;
  const auto device_id = 0;

  // Example arguments
  const size_t batch_count = 261;
  const size_t m = 1;
  const size_t n = 1;
  const size_t k = 40;
  const auto a_ld = 2560;
  const auto b_ld = 160;
  const auto c_ld = 261;
  std::vector<float> alphas(batch_count);
  std::vector<float> betas(batch_count);
  std::vector<size_t> a_offsets(batch_count);
  std::vector<size_t> b_offsets(batch_count);
  std::vector<size_t> c_offsets(batch_count);
  for (auto b_id = size_t{0}; b_id < batch_count; ++b_id) {
    alphas[b_id] = 1.0f;
    betas[b_id] = 1.0f;
    a_offsets[b_id] = 0;
    b_offsets[b_id] = 0;
    c_offsets[b_id] = b_id;
  }
  const auto a_size = a_ld * m;
  const auto b_size = b_ld * k;
  const auto c_size = c_ld * k;

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
  auto host_a = std::vector<float>(a_size);
  auto host_b = std::vector<float>(b_size);
  auto host_c = std::vector<float>(c_size);
  for (auto &item: host_a) { item = 12.193f; }
  for (auto &item: host_b) { item = -8.199f; }
  for (auto &item: host_c) { item = 0.0f; }

  // Copy the matrices to the device
  auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, host_a.size()*sizeof(float));
  auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, host_b.size()*sizeof(float));
  auto device_c = cl::Buffer(context, CL_MEM_READ_WRITE, host_c.size()*sizeof(float));
  queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size()*sizeof(float), host_a.data());
  queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size()*sizeof(float), host_b.data());
  queue.enqueueWriteBuffer(device_c, CL_TRUE, 0, host_c.size()*sizeof(float), host_c.data());

  // Start the timer
  auto start_time = std::chrono::steady_clock::now();

  // Calls the routine. Note that the type of alphas and betas (float) determine the precision.
  auto queue_plain = queue();
  auto status = clblast::GemmBatched(clblast::Layout::kRowMajor,
                                     clblast::Transpose::kNo, clblast::Transpose::kNo,
                                     m, n, k,
                                     alphas.data(),
                                     device_a(), a_offsets.data(), a_ld,
                                     device_b(), b_offsets.data(), b_ld,
                                     betas.data(),
                                     device_c(), c_offsets.data(), c_ld,
                                     batch_count,
                                     &queue_plain, &event);

  // Record the execution time
  if (status == clblast::StatusCode::kSuccess) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }
  auto elapsed_time = std::chrono::steady_clock::now() - start_time;
  auto time_ms = std::chrono::duration<double,std::milli>(elapsed_time).count();

  // Example completed. See "clblast.h" for status codes (0 -> success).
  printf("Completed batched SGEMM in %.3lf ms with status %d\n", time_ms, static_cast<int>(status));
  return 0;
}

// =================================================================================================
