
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the HAXPY routine. It demonstrates the use of half-precision.
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

// Includes the float-to-half and half-to-float conversion utilities
#include <clblast_half.h>

// =================================================================================================

// Example use of the half-precision routine HAXPY
int main(void) {

  // OpenCL platform/device settings
  const size_t platform_id = 0;
  const size_t device_id = 0;

  // Example HAXPY arguments
  const size_t n = 8192;
  const cl_half alpha = FloatToHalf(0.5f);

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

  // Populate host vectors with some example data
  cl_half* host_a = (cl_half*)malloc(sizeof(cl_half)*n);
  cl_half* host_b = (cl_half*)malloc(sizeof(cl_half)*n);
  for (size_t i=0; i<n; ++i) { host_a[i] = FloatToHalf(2.2f); }
  for (size_t i=0; i<n; ++i) { host_b[i] = FloatToHalf(0.4f); }
  printf("Input values at index 0: alpha * a[0] + b[0] == %.3lf * %.3lf + %.3lf\n",
         HalfToFloat(alpha), HalfToFloat(host_a[0]), HalfToFloat(host_b[0]));

  // Copy the matrices to the device
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(cl_half), NULL, NULL);
  cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(cl_half), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, n*sizeof(cl_half), host_a, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n*sizeof(cl_half), host_b, 0, NULL, NULL);

  // Call the HAXPY routine.
  CLBlastStatusCode status = CLBlastHaxpy(n, alpha,
                                          device_a, 0, 1,
                                          device_b, 0, 1,
                                          &queue, &event);

  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }

  // Copies the result back to the host
  clEnqueueReadBuffer(queue, device_b, CL_TRUE, 0, n*sizeof(cl_half), host_b, 0, NULL, NULL);

  // Example completed. See "clblast_c.h" for status codes (0 -> success).
  printf("Completed HAXPY with status %d\n", status);

  // Prints the first output value
  if (status == 0) {
    printf("Output value at index 0: b[0] = %.3lf\n", HalfToFloat(host_b[0]));
  }

  // Clean-up
  free(platforms);
  free(devices);
  free(host_a);
  free(host_b);
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}

// =================================================================================================
