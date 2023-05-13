
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the SGEMM routine. It is pure C99 and demonstrates the use of
// the C API to the CLBlast library.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
//
// =================================================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the CLBlast library (C interface)
#include <clblast_c.h>

// =================================================================================================

// Example use of the single-precision routine SGEMM
int main(void) {

  // OpenCL platform/device settings
  const size_t platform_id = 0;
  const size_t device_id = 0;

  // Example SGEMM arguments
  const size_t m = 128;
  const size_t n = 64;
  const size_t k = 512;
  const float alpha = 0.7f;
  const float beta = 1.0f;
  const size_t a_ld = k;
  const size_t b_ld = n;
  const size_t c_ld = n;

  // Initializes the OpenCL platform
  cl_uint num_platforms;
  clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
  clGetPlatformIDs(num_platforms, platforms, NULL);
  cl_platform_id platform = platforms[platform_id];

  // Initializes the OpenCL device
  cl_uint num_devices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
  cl_device_id device = devices[device_id];

  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
  cl_event event = NULL;

  // Populate host matrices with some example data
  float* host_a = (float*)malloc(sizeof(float)*m*k);
  float* host_b = (float*)malloc(sizeof(float)*n*k);
  float* host_c = (float*)malloc(sizeof(float)*m*n);
  for (size_t i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
  for (size_t i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
  for (size_t i=0; i<m*n; ++i) { host_c[i] = 0.0f; }

  // Copy the matrices to the device
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*k*sizeof(float), NULL, NULL);
  cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*k*sizeof(float), NULL, NULL);
  cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(float), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m*k*sizeof(float), host_a, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n*k*sizeof(float), host_b, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, m*n*sizeof(float), host_c, 0, NULL, NULL);

  // Call the SGEMM routine.
  CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                          CLBlastTransposeNo, CLBlastTransposeNo,
                                          m, n, k,
                                          alpha,
                                          device_a, 0, a_ld,
                                          device_b, 0, b_ld,
                                          beta,
                                          device_c, 0, c_ld,
                                          &queue, &event);

  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }

  // Example completed. See "clblast_c.h" for status codes (0 -> success).
  printf("Completed SGEMM with status %d\n", status);

  // Clean-up
  free(platforms);
  free(devices);
  free(host_a);
  free(host_b);
  free(host_c);
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseMemObject(device_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}

// =================================================================================================
