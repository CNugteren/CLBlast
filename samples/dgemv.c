
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the DGEMV routine. It is pure C99 and demonstrates the use of
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

// Example use of the double-precision routine DGEMV
int main(void) {

  // OpenCL platform/device settings
  const size_t platform_id = 0;
  const size_t device_id = 0;

  // Example DGEMV arguments
  const size_t m = 128;
  const size_t n = 289;
  const double alpha = 0.7;
  const double beta = 0.0;
  const size_t a_ld = n;

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

  // Populate host data structures with some example data
  double* host_a = (double*)malloc(sizeof(double)*m*n);
  double* host_x = (double*)malloc(sizeof(double)*n);
  double* host_y = (double*)malloc(sizeof(double)*m);
  for (size_t i=0; i<m*n; ++i) { host_a[i] = 12.193; }
  for (size_t i=0; i<n; ++i) { host_x[i] = -8.199; }
  for (size_t i=0; i<m; ++i) { host_y[i] = 0.0; }

  // Copy the data-structures to the device
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(double), NULL, NULL);
  cl_mem device_x = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(double), NULL, NULL);
  cl_mem device_y = clCreateBuffer(context, CL_MEM_READ_WRITE, m*sizeof(double), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m*n*sizeof(double), host_a, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_x, CL_TRUE, 0, n*sizeof(double), host_x, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_y, CL_TRUE, 0, m*sizeof(double), host_y, 0, NULL, NULL);

  // Call the DGEMV routine.
  CLBlastStatusCode status = CLBlastDgemv(CLBlastLayoutRowMajor, CLBlastTransposeNo,
                                          m, n,
                                          alpha,
                                          device_a, 0, a_ld,
                                          device_x, 0, 1,
                                          beta,
                                          device_y, 0, 1,
                                          &queue, &event);

  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }

  // Example completed. See "clblast_c.h" for status codes (0 -> success).
  printf("Completed DGEMV with status %d\n", status);

  // Clean-up
  free(platforms);
  free(devices);
  free(host_a);
  free(host_x);
  free(host_y);
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_x);
  clReleaseMemObject(device_y);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}

// =================================================================================================
