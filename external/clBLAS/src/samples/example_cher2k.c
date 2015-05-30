/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#include <sys/types.h>
#include <stdio.h>
#include <string.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */
static const clblasOrder order = clblasColumnMajor;
static const clblasUplo uplo = clblasLower;
static const clblasTranspose transA = clblasNoTrans;

static const size_t N = 5;
static const size_t K = 4;

static const cl_float2 alpha = {{10, 1}};
static const cl_float2 A[] = {

    {{11, 0}}, {{12, 0}}, {{13, 0}}, {{14, 0}},
    {{21, 0}}, {{22, 0}}, {{23, 0}}, {{24, 0}},
    {{31, 0}}, {{32, 0}}, {{33, 0}}, {{34, 0}},
    {{41, 0}}, {{42, 0}}, {{43, 0}}, {{44, 0}},
    {{51, 0}}, {{52, 0}}, {{53, 0}}, {{54, 0}}
};
static const size_t lda = 5;    /* i.e. lda = N */

static const cl_float2 B[] = {

    {{1, 0}}, {{2, 0}}, {{3, 0}}, {{4, 0}},
    {{2, 0}}, {{2, 0}}, {{3, 0}}, {{4, 0}},
    {{3, 0}}, {{2, 0}}, {{3, 0}}, {{3, 0}},
    {{4, 0}}, {{4, 0}}, {{4, 0}}, {{4, 0}},
    {{5, 0}}, {{5, 0}}, {{5, 0}}, {{5, 0}}
};
static const size_t ldb = 5;    /* i.e. lda = N */

static const cl_float beta = 1;
static cl_float2 C[] = {
    {{11, 1}}, {{12, 0}}, {{13, 0}}, {{14, 0}}, {{15, 0}},
    {{ 0, 0}}, {{22, 2}}, {{23, 0}}, {{24, 0}}, {{25, 0}},
    {{ 0, 0}}, {{ 0, 0}}, {{33, 4}}, {{34, 0}}, {{35, 0}},
    {{ 0, 0}}, {{ 0, 0}}, {{ 0, 0}}, {{44, 5}}, {{45, 0}},
    {{ 0, 0}}, {{ 0, 0}}, {{ 0, 0}}, {{ 0, 0}}, {{55, 6}}
};
static const size_t ldc = 5;    /* i.e. ldc = N */

static void
printResult(void)
{
    size_t i, j;

    printf("Result:\n");

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("(%9.2f, %-9.2f) ", CREAL(C[i + j * ldc]), CIMAG(C[i + j * ldc]));
        }
        printf("\n");
    }
}

int
main(void)
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufC, bufB;
    cl_event event = NULL;
    int ret = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * K * sizeof(*A),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * K * sizeof(*B),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * N * sizeof(*C),
                          NULL, &err);

    if ((bufA == NULL) || (bufC == NULL) || (bufB == NULL))
    {
        printf("Failed to create buffern");
        return 1;
    }
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        N * K * sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        N * K * sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
        N * N * sizeof(*C), C, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasCher2k(order, uplo, transA, N, K, alpha, bufA, 0, lda, bufB, 0, ldb,
                            beta, bufC, 0, ldc, 1, &queue, 0, NULL, &event);

    if (err != CL_SUCCESS) {
        printf("clblasCher2k() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, N * N * sizeof(*C),
                                  C, 0, NULL, NULL);

        /* At this point you will get the result of SSYRK placed in C array. */
        printResult();
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
