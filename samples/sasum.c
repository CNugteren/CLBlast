
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the SASUM routine. It is pure C99 and demonstrates the use of
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

// Example use of the single-precision routine SASUM
int main(void) {

  // OpenCL platform/device settings
  const size_t platform_id = 0;
  const size_t device_id = 0;

  // Example SASUM arguments
  const size_t n = 1000;
  const float input_value = -1.5f;

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
  float* host_input = (float*)malloc(sizeof(float)*n);
  float* host_output = (float*)malloc(sizeof(float)*1);
  for (size_t i=0; i<n; ++i) { host_input[i] = input_value; }
  for (size_t i=0; i<1; ++i) { host_output[i] = 0.0f; }

  // Copy the data-structures to the device
  cl_mem device_input = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, NULL);
  cl_mem device_output = clCreateBuffer(context, CL_MEM_READ_WRITE, 1*sizeof(float), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_input, CL_TRUE, 0, n*sizeof(float), host_input, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_output, CL_TRUE, 0, 1*sizeof(float), host_output, 0, NULL, NULL);

  // Call the SASUM routine.
  CLBlastStatusCode status = CLBlastSasum(n,
                                          device_output, 0,
                                          device_input, 0, 1,
                                          &queue, &event);

  // Wait for completion
  if (status == CLBlastSuccess) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }

  // Copies the result back to the host
  clEnqueueReadBuffer(queue, device_output, CL_TRUE, 0, 1*sizeof(float), host_output, 0, NULL, NULL);

  // Example completed. See "clblast_c.h" for status codes (0 -> success).
  printf("Completed SASUM with status %d: %zu * |%.1lf| = %.1lf\n", status, n, input_value, host_output[0]);

  // Clean-up
  free(platforms);
  free(devices);
  free(host_input);
  free(host_output);
  clReleaseMemObject(device_input);
  clReleaseMemObject(device_output);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}

// =================================================================================================
