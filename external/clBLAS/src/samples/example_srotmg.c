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
#include <math.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */
static cl_float SD1 = 10;
static cl_float SD2 = 21;
static cl_float SX1 = 1;
static cl_float SY1 = -1;
static cl_float SPARAM[] = {
    -1,
    10,
    12,
    20,
    2
};

static void
printResult(void)
{
    printf("\nResult:\n");
    printf("SD1: %f,\tSD2: %f,\t SX1: %f,\tSY1: %f\nSPARAM: %f %f %f %f %f\n",
            SD1, SD2, SX1, SY1, SPARAM[0], SPARAM[1], SPARAM[2], SPARAM[3], SPARAM[4]);
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
    cl_mem bufD1, bufD2, bufX1, bufY1, bufParam;
    cl_event event = NULL;
    int ret = 0;
    int lenParam = 5;

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
    bufD1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    bufD2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    bufX1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    bufY1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    bufParam = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (lenParam*sizeof(cl_float)), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufD1, CL_TRUE, 0, sizeof(cl_float), &SD1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufD2, CL_TRUE, 0, sizeof(cl_float), &SD2, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufX1, CL_TRUE, 0, sizeof(cl_float), &SX1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY1, CL_TRUE, 0, sizeof(cl_float), &SY1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufParam, CL_TRUE, 0, (lenParam*sizeof(cl_float)), SPARAM, 0, NULL, NULL);


    /* Call clblas function. */
    err = clblasSrotmg(bufD1, 0, bufD2, 0, bufX1, 0, bufY1, 0, bufParam, 0, 1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSrotmg() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufD1, CL_TRUE, 0, sizeof(cl_float), &SD1, 0, NULL, NULL);
        err = clEnqueueReadBuffer(queue, bufD2, CL_TRUE, 0, sizeof(cl_float), &SD2, 0, NULL, NULL);
        err = clEnqueueReadBuffer(queue, bufX1, CL_TRUE, 0, sizeof(cl_float), &SX1, 0, NULL, NULL);
        err = clEnqueueReadBuffer(queue, bufY1, CL_TRUE, 0, sizeof(cl_float), &SY1, 0, NULL, NULL);
        err = clEnqueueReadBuffer(queue, bufParam, CL_TRUE, 0,
                        (lenParam*sizeof(cl_float)), SPARAM, 0, NULL, NULL);

        /* At this point you will get the result of SROTG placed in vector Y. */
        printResult();
    }

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufD1);
    clReleaseMemObject(bufD2);
    clReleaseMemObject(bufX1);
    clReleaseMemObject(bufY1);
    clReleaseMemObject(bufParam);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
