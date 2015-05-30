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


#include <stdio.h>
#include <clkern.h>
#include <stdlib.h>
#include <trace_malloc.h>

enum {
    MAX_SOURCE_SIZE = 1048576,
    MAX_OPENCL_DEVICES = 64
};

static size_t
getBinSizeAndIdx(cl_program program, int *idx)
{
    size_t allSizes[MAX_OPENCL_DEVICES], size = 0;
    size_t i, retSize;

    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                     sizeof(allSizes), &allSizes, &retSize);
    retSize /= sizeof(size);
    for (i = 0; i < retSize; i++) {
        size = allSizes[i];
        if (size) {
            break;
        }
    }

    if (idx && (i < retSize)) {
        *idx = (int)i;
    }

    return size;
}

cl_int
launchClKernel(
    KernelDesc *kernDesc,
    cl_command_queue queue,
    KernelErrorInfo *errInfo)
{
    cl_int status;
    unsigned int i;
    KernelArg *karg;
    KernelErrorInfo ei;
    unsigned long t;
    unsigned int nrArgs;

    errInfo->phase = -1;
    errInfo->wrongArg = (unsigned int)-1;
    ei.phase = -1;
    ei.wrongArg = (unsigned int)-1;

    status = clGetKernelInfo(kernDesc->kernel, CL_KERNEL_NUM_ARGS,
                             sizeof(nrArgs), &nrArgs, NULL);
    if (status != CL_SUCCESS) {
        return status;
    }

    karg = kernDesc->args;
    for (i = 0; (i < nrArgs) && (status == CL_SUCCESS); i++, karg++) {
        status = clSetKernelArg(kernDesc->kernel, i, karg->typeSize,
                                karg->arg.data);
        if (status != CL_SUCCESS) {
            ei.wrongArg = i;
            ei.phase = PHASE_SET_ARGS;
        }
        else if (karg->hostBuf && (karg->dir & MEMOBJ_WRITE)) {
            status = clEnqueueWriteBuffer(queue, karg->arg.mem,
                                          CL_TRUE, 0, karg->hostBufLen,
                                          karg->hostBuf, 0, NULL, NULL);
            if (status != CL_SUCCESS) {
                ei.wrongArg = i;
                ei.phase = PHASE_ENQUEUE_WRITE;
            }
        }
    }

    if (status == CL_SUCCESS) {
        status = clEnqueueNDRangeKernel(queue,
                                        kernDesc->kernel,
                                        (cl_uint)kernDesc->workDim,
                                        NULL,
                                        (const size_t*)kernDesc->globalThreads,
                                        (const size_t*)kernDesc->localThreads,
                                        (cl_uint)kernDesc->waitListSize,
                                        kernDesc->eventWaitList,
                                        kernDesc->event);
        if ((status == CL_SUCCESS) && !kernDesc->nowait) {
            status = clWaitForEvents(1, kernDesc->event);
        }

        if (status != CL_SUCCESS) {
            ei.phase = PHASE_ENQUEUE_KERNEL;
        }

        if ((status == CL_SUCCESS) && kernDesc->needExecTime &&
            kernDesc->event) {

            if (kernDesc->nowait) {
                status = clWaitForEvents(1, kernDesc->event);
                if (status != CL_SUCCESS) {
                    ei.phase = PHASE_PROFILING;
                }
            }

            if (status == CL_SUCCESS) {
                status = clGetEventProfilingInfo(*kernDesc->event,
                                                 CL_PROFILING_COMMAND_START,
                                                 sizeof(t), &t, NULL);
                if (status == CL_SUCCESS) {
                    status = clGetEventProfilingInfo(*kernDesc->event,
                                                     CL_PROFILING_COMMAND_END,
                                                     sizeof(kernDesc->execTime),
                                                     &kernDesc->execTime, NULL);
                    kernDesc->execTime -= t;
                }
                if (status != CL_SUCCESS) {
                    ei.phase = PHASE_PROFILING;
                }
            }
        }
    }

    karg = kernDesc->args;
    for (i = 0; (i < nrArgs) && (status == CL_SUCCESS); i++, karg++) {
        if (karg->hostBuf && (karg->dir & MEMOBJ_READ)) {
            status = clEnqueueReadBuffer(queue, karg->arg.mem,
                                         CL_TRUE, 0, karg->hostBufLen,
                                         karg->hostBuf, 0, NULL, NULL);
            if (status != CL_SUCCESS) {
                ei.wrongArg = i;
                ei.phase = PHASE_ENQUEUE_READ;
            }
        }
    }

    if ((status != CL_SUCCESS) && errInfo) {
        errInfo->phase = ei.phase;
        if (ei.phase != PHASE_ENQUEUE_KERNEL) {
            errInfo->wrongArg = ei.wrongArg;
        }
    }

    return status;
}

cl_program
buildClProgram(
    const char *source,
    const char *buildOpts,
    cl_context ctx,
    cl_device_id devID,
    char *logBuf,
    size_t logBufSize,
    cl_int *status)
{
    cl_program program = NULL;
    cl_int stat = CL_SUCCESS;

    program = clCreateProgramWithSource(ctx, 1, (const char**)&source,
                                        NULL, &stat);
    if (program != NULL) {
        stat = clBuildProgram(program, 1, (const cl_device_id*)&devID,
                              buildOpts, NULL, NULL);
        if (stat != CL_SUCCESS) {
            if (logBuf) {
                logBuf[0] = '\0';
                clGetProgramBuildInfo(program, devID,
                                      CL_PROGRAM_BUILD_LOG,
                                      logBufSize, logBuf, NULL);
            }
            clReleaseProgram(program);
            program = NULL;
        }
    }

    if (status) {
        *status = stat;
    }

    return program;
}

cl_program
createClProgramWithBinary(
    cl_context ctx,
    cl_device_id devID,
    unsigned char *binary,
    size_t binSize,
    cl_int *status)
{
    cl_program program;
    cl_int s;

    program = clCreateProgramWithBinary(ctx, 1, &devID, &binSize,
                                        (const unsigned char**)&binary,
                                        NULL, &s);
    if (program != NULL) {
        s = clBuildProgram(program, 1, &devID, NULL, NULL, NULL);
        if (s != CL_SUCCESS) {
            clReleaseProgram(program);
            program = NULL;
        }
    }

    if (status != NULL) {
        *status = s;
    }

    return program;
}

size_t
getProgramBinarySize(cl_program program)
{
    return getBinSizeAndIdx(program, NULL);
}

unsigned char
*getProgramBinary(cl_program program)
{
    unsigned char *binaries[MAX_OPENCL_DEVICES];
    unsigned char *bin = NULL;
    size_t size;
    int idx = 0;

    memset(binaries, 0, sizeof(binaries));
    size = getBinSizeAndIdx(program, &idx);
    bin = binaries[idx] = malloc(size);
    if (bin != NULL) {
        cl_int err;

        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binaries),
                               binaries, NULL);
        if (err != CL_SUCCESS) {
            free(bin);
            bin = NULL;
        }
    }

    return bin;
}
