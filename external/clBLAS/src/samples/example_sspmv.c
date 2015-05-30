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
static const cl_float alpha = 10;

static const clblasUplo uplo = clblasUpper;
static const cl_float AP[] = {
    11, 12, 13, 14, 15,
        22, 23, 24, 25,
            33, 34, 35,
                44, 45,
                    55
};
static const size_t lda = 5;    /* i.e. lda = N */

static const cl_float X[] = {
    11,
    21,
    31,
    41,
    51
};
static const int incx = 1;

static const cl_float beta = 20;
static cl_float Y[] = {
    11,
    21,
    31,
    41,
    51
};
static const int incy = 1;

static void
printResult(void)
{
    size_t i;

    printf("Result:\n\n");
    for (i = 0; i < N; i++) {
        printf("%.3f\n", Y[i * incy]);
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
    cl_mem bufAP, bufX, bufY;
    cl_event event = NULL;
    int ret = 0, numElementsAP;

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

    numElementsAP = (N * (N+1)) / 2;	// To get number of elements in a packed matrix

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufAP = clCreateBuffer(ctx, CL_MEM_READ_ONLY, numElementsAP * sizeof(cl_float),
                          NULL, &err);
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(cl_float),
                          NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(cl_float),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufAP, CL_TRUE, 0,
                        numElementsAP * sizeof(cl_float), AP, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
                        N * sizeof(cl_float), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
                        N * sizeof(cl_float), Y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasSspmv(order, uplo, N, alpha, bufAP, 0, bufX, 0, incx,
                                beta, bufY, 0, incy, 1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSspmv() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, N * sizeof(cl_float),
                                            Y, 0, NULL, NULL);
        /* At this point you will get the result of SSPMV placed in Y array. */
        printResult();
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufAP);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
