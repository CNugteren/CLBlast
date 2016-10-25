
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the Netlib CBLAS API of the CLBlast library. This API is not
// recommended if you want full control over performance: it will internally copy buffers from and
// to the OpenCL device.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
//
// =================================================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Includes the CLBlast library (Netlib CBLAS interface)
#include <clblast_netlib_c.h>

// =================================================================================================

// Example use of the single-precision routine SGEMM
int main(void) {

  // Example SGEMM arguments
  const int m = 128;
  const int n = 64;
  const int k = 512;
  const float alpha = 0.7f;
  const float beta = 1.0f;
  const int a_ld = k;
  const int b_ld = n;
  const int c_ld = n;

  // Populate host matrices with some example data
  float* host_a = (float*)malloc(sizeof(float)*m*k);
  float* host_b = (float*)malloc(sizeof(float)*n*k);
  float* host_c = (float*)malloc(sizeof(float)*m*n);
  for (int i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
  for (int i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
  for (int i=0; i<m*n; ++i) { host_c[i] = 0.0f; }

  // Call the SGEMM routine.
  cblas_sgemm(CLBlastLayoutRowMajor,
              CLBlastTransposeNo, CLBlastTransposeNo,
              m, n, k,
              alpha,
              host_a, a_ld,
              host_b, b_ld,
              beta,
              host_c, c_ld);

  // Example completed
  printf("Completed SGEMM\n");

  // Clean-up
  free(host_a);
  free(host_b);
  free(host_c);
  return 0;
}

// =================================================================================================
