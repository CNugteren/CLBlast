
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the SGEMM routine with the C++ CUDA API of CLBlast.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
//
// =================================================================================================

#include <cstdio>
#include <chrono>
#include <vector>

// Includes the CUDA driver API
#include <cuda.h>

// Includes the CLBlast library
#include <clblast_cuda.h>

// =================================================================================================

// Example use of the single-precision Xgemm routine SGEMM
int main() {

  // CUDA device selection
  const auto device_id = 0;

  // Example SGEMM arguments
  const size_t m = 128;
  const size_t n = 64;
  const size_t k = 512;
  const float alpha = 0.7f;
  const float beta = 1.0f;
  const auto a_ld = k;
  const auto b_ld = n;
  const auto c_ld = n;

  // Initializes the OpenCL device
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, device_id);

  // Creates the OpenCL context and stream
  CUcontext context;
  cuCtxCreate(&context, 0, device);
  CUstream stream;
  cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

  // Populate host matrices with some example data
  auto host_a = std::vector<float>(m*k);
  auto host_b = std::vector<float>(n*k);
  auto host_c = std::vector<float>(m*n);
  for (auto &item: host_a) { item = 12.193f; }
  for (auto &item: host_b) { item = -8.199f; }
  for (auto &item: host_c) { item = 0.0f; }

  // Copy the matrices to the device
  CUdeviceptr device_a;
  CUdeviceptr device_b;
  CUdeviceptr device_c;
  cuMemAlloc(&device_a, host_a.size()*sizeof(float));
  cuMemAlloc(&device_b, host_b.size()*sizeof(float));
  cuMemAlloc(&device_c, host_c.size()*sizeof(float));
  cuMemcpyHtoDAsync(device_a, host_a.data(), host_a.size()*sizeof(float), stream);
  cuMemcpyHtoDAsync(device_b, host_b.data(), host_b.size()*sizeof(float), stream);
  cuMemcpyHtoDAsync(device_c, host_c.data(), host_c.size()*sizeof(float), stream);

  // Start the timer
  auto start_time = std::chrono::steady_clock::now();

  // Call the SGEMM routine. Note that the type of alpha and beta (float) determine the precision.
  auto status = clblast::Gemm(clblast::Layout::kRowMajor,
                              clblast::Transpose::kNo, clblast::Transpose::kNo,
                              m, n, k,
                              alpha,
                              device_a, 0, a_ld,
                              device_b, 0, b_ld,
                              beta,
                              device_c, 0, c_ld,
                              context, device);
  cuStreamSynchronize(stream);

  // Record the execution time
  auto elapsed_time = std::chrono::steady_clock::now() - start_time;
  auto time_ms = std::chrono::duration<double,std::milli>(elapsed_time).count();

  // Example completed. See "clblast_cuda.h" for status codes (0 -> success).
  printf("Completed SGEMM in %.3lf ms with status %d\n", time_ms, static_cast<int>(status));

  // Clean-up
  cuMemFree(device_a);
  cuMemFree(device_b);
  cuMemFree(device_c);
  cuStreamDestroy(stream);
  return 0;
}

// =================================================================================================
