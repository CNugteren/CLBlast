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
static cl_double2 alpha = {{10.0f, 2.0f}};
static const clblasUplo uplo = clblasUpper;

static cl_double2 A[] = {
    {{11.0f, 00.0f}}, {{12.0f, 02.0f}}, {{13.0f, 05.0f}}, {{14.0f, 12.0f}}, {{15.0f, 06.0f}},
    {{00.0f, 00.0f}}, {{22.0f, 00.0f}}, {{23.0f, 25.0f}}, {{24.0f, 23.0f}}, {{25.0f, 61.0f}},
    {{00.0f, 00.0f}}, {{00.0f, 00.0f}}, {{33.0f, 00.0f}}, {{34.0f, 23.0f}}, {{35.0f, 21.0f}},
    {{00.0f, 00.0f}}, {{00.0f, 00.0f}}, {{00.0f, 00.0f}}, {{44.0f, 00.0f}}, {{45.0f, 23.0f}},
    {{00.0f, 00.0f}}, {{00.0f, 00.0f}}, {{00.0f, 00.0f}}, {{00.0f, 00.0f}}, {{55.0f, 00.0f}}
};
static const size_t lda = 5;    /* i.e. lda = N */

static const cl_double2 X[] = {
    {{11.0f, 03.0f}},
    {{01.0f, 15.0f}},
    {{30.0f, 20.0f}},
    {{01.0f, 02.0f}},
    {{11.0f, 10.0f}}
};
static const int incx = 1;

static const cl_double2 Y[] = {
    {{11.0f, 03.0f}},
    {{03.0f, 05.0f}},
    {{09.0f, 00.0f}},
    {{01.0f, 02.0f}},
    {{11.0f, 00.0f}}
};
static const int incy = 1;



static void
printResult(void)
{
    size_t i, j;
    printf("\nResult:\n");

    for (i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			printf("(%9.2lf, %-9.2lf)\t", CREAL( A[ i*N + j ] ), CIMAG( A[ i*N + j ] ));
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
    cl_mem bufA, bufX, bufY;
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
    bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * lda * sizeof(cl_double2),
                          NULL, &err);
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(cl_double2),
                          NULL, &err);
	bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(cl_double2),
						  NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
					N * lda * sizeof(cl_double2), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
					N * sizeof(cl_double2), X, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0,
					N * sizeof(cl_double2), Y, 0, NULL, NULL);

    err = clblasZher2(order, uplo, N, alpha, bufX, 0 /*offx */, incx, bufY, 0 /*offy*/, incy,
						bufA, 0 /*offa */, lda, 1, &queue, 0, NULL, &event);

   	if (err != CL_SUCCESS) {
        printf("clblasZher2() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, (N * lda * sizeof(cl_double2)),
                                  A, 0, NULL, NULL);
        /* At this point you will get the result of ZHER2 placed in A array. */
        printResult();
    }


    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufA);
	clReleaseMemObject(bufY);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
