
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the DAXPY routine with the C++ CUDA API of CLBlast.
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

// Example use of the double-precision Xaxpy routine DAXPY
int main() {

  // CUDA device selection
  const auto device_id = 0;

  // Example DAXPY arguments
  const size_t n = 8192;
  const double alpha = 0.7;

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
  auto host_a = std::vector<double>(n);
  auto host_b = std::vector<double>(n);
  for (auto &item: host_a) { item = 12.193; }
  for (auto &item: host_b) { item = -8.199; }

  // Copy the matrices to the device
  CUdeviceptr device_a;
  CUdeviceptr device_b;
  cuMemAlloc(&device_a, host_a.size()*sizeof(double));
  cuMemAlloc(&device_b, host_b.size()*sizeof(double));
  cuMemcpyHtoDAsync(device_a, host_a.data(), host_a.size()*sizeof(double), stream);
  cuMemcpyHtoDAsync(device_b, host_b.data(), host_b.size()*sizeof(double), stream);

  // Start the timer
  auto start_time = std::chrono::steady_clock::now();

  // Call the DAXPY routine. Note that the type of alpha (double) determines the precision.
  const auto status = clblast::Axpy(n, alpha,
                                    device_a, 0, 1,
                                    device_b, 0, 1,
                                    context, device);
  cuStreamSynchronize(stream);

  // Record the execution time
  auto elapsed_time = std::chrono::steady_clock::now() - start_time;
  auto time_ms = std::chrono::duration<double,std::milli>(elapsed_time).count();

  // Example completed. See "clblast_cuda.h" for status codes (0 -> success).
  printf("Completed DAXPY in %.3lf ms with status %d\n", time_ms, static_cast<int>(status));

  // Clean-up
  cuMemFree(device_a);
  cuMemFree(device_b);
  cuStreamDestroy(stream);
  return 0;
}

// =================================================================================================
