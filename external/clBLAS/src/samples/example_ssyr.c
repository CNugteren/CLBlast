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
static const clblasOrder order = clblasRowMajor;

static const size_t N = 5;

static cl_float alpha = 10.0;

static const clblasUplo uplo = clblasUpper;
static cl_float A[] = {
    1.0, 2.0,  3.0,  4.0,  5.0,
    0.0, 6.0,  7.0,  8.0,  9.0,
    0.0, 0.0, 10.0, 11.0, 12.0,
    0.0, 0.0,  0.0, 13.0, 14.0,
    0.0, 0.0,  0.0, 00.0, 15.0
};
static const size_t lda = 5;    /* i.e. lda = N */

static const cl_float X[] = {
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
};
static const int incx = 1;

static void
printResult(void)
{
    size_t i, j;
    printf("\nResult:\n");

    for (i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			printf("\t(%.2f)", A[ i*N + j ]);
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
    cl_mem bufA, bufX;
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

	printResult();

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * lda * sizeof(cl_float),
                          NULL, &err);
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(cl_float),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        N * lda * sizeof(cl_float), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
        N * sizeof(cl_float), X, 0, NULL, NULL);


    err = clblasSsyr(order, uplo, N, alpha, bufX, 0 /*offx */, incx,
						bufA, 0 /*offa */, lda, 1, &queue, 0, NULL, &event);

   	if (err != CL_SUCCESS) {
        printf("clblasSsyr() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, (N * lda * sizeof(cl_float)),
                                  A, 0, NULL, NULL);
        /* At this point you will get the result of SSYR placed in A array. */
        printResult();
    }


    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufA);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
