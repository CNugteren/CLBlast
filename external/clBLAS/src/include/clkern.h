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


#ifndef CLARGS_H_
#define CLARGS_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string.h>
#include <dis_warning.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INIT_KARG(karg, val)                            \
do {                                                    \
    memcpy((karg)->arg.data, &(val), sizeof(val));      \
    (karg)->typeSize = sizeof(val);                     \
} while (0)

enum {
    MAX_KERNEL_ARGS = 32,
    MAX_ARG_SIZE = sizeof(cl_double2),
    MAX_WORK_DIM = 3
};

// memory object data transfer direction
typedef enum MemobjDir {
    MEMOBJ_READ = 0x1,
    MEMOBJ_WRITE = 0x2
} MemobjDir;

typedef enum KernelLaunchPhase {
    PHASE_SET_ARGS,
    PHASE_ENQUEUE_WRITE,
    PHASE_ENQUEUE_KERNEL,
    PHASE_PROFILING,
    PHASE_ENQUEUE_READ
} KernelLaunchPhase;

typedef union KernelArgValue {
    cl_mem mem;
    int ival;
    unsigned char data[MAX_ARG_SIZE];
} KernelArgValue;

/*
 * Structure describing an argument to be passed to a kernel
 *
 * @arg:        pointer to the argument
 * @ardIdx:     argument index in the kernel argument list
 * @hostBuf:    buffer to copy data to/from from/to GPU memory
 * @enqType:    buffer enqueue type
 * @sync:       blocking I/O
 * @event:      event for I/O
 */
typedef struct KernelArg {
    KernelArgValue arg;
    unsigned int typeSize;  // argument type size, ignored for mem objects
    void *hostBuf;          // host buffer for using with OpenCL memory objects
    size_t hostBufLen;
    MemobjDir dir;
} KernelArg;

typedef struct KernelDesc {
    cl_kernel kernel;
    size_t globalThreads[MAX_WORK_DIM];
    size_t localThreads[MAX_WORK_DIM];
    size_t workDim;
    const cl_event *eventWaitList;
    size_t waitListSize;
    cl_event *event;
    int nowait;
    int needExecTime;
    KernelArg args[MAX_KERNEL_ARGS];
    unsigned long execTime;
} KernelDesc;

typedef struct KernelErrorInfo {
    unsigned int wrongArg;
    KernelLaunchPhase phase;
} KernelErrorInfo;

/*
 * store kernel arguments launch the kernel and read its results
 *
 * @kernDesc:    descriptor of the kernel to be launched
 * @queue:       command queue associated with the device
 * @errInfo:     location to store info about occurred error,
 *               ignored if NULL
 *
 * The function gets itself number of arguments to the kernel
 * usging the OpenCL API
 */
cl_int launchClKernel(
    KernelDesc *kernDesc,
    cl_command_queue queue,
    KernelErrorInfo *errInfo);

/*
 * build a program from source
 *
 * @source:     program source
 * @buildOpts:  options to the opencl program builder
 * @DevID:      ID of device to create program for
 * @logBuf:     buffer to store build log at error
 * @status:     location to store OpenCL status at error
 *
 * On success returns a build program object.
 * On error returns <NULL>, and stores to the 'status' location
 * opencl status; if <NULL> result is returned, but 'status'
 * cointains 'CL_SUCCESS', it means an file I/O or memory allocation
 * failure is occurred. If 'status' is set to NULL, it is ignored
 */
cl_program
buildClProgram(
    const char *source,
    const char *buildOpts,
    cl_context ctx,
    cl_device_id devID,
    char *logBuf,
    size_t logBufSize,
    cl_int *status);

/*
 * TODO: Doxygen-style comments
 */
cl_program
createClProgramWithBinary(
    cl_context ctx,
    cl_device_id devID,
    unsigned char *binary,
    size_t binSize,
    cl_int *status);

/*
 * TODO: Doxygen-style comments
 */
size_t
getProgramBinarySize(cl_program program);

/*
 * TODO: Doxygen-style comments
 */
unsigned char
*getProgramBinary(cl_program program);

/*
 * set a kernel argument of the size_t type
 */
static __inline void
initSizeKarg(KernelArg *arg, size_t value)
{
    memcpy(arg->arg.data, &value, sizeof(cl_uint));
    arg->typeSize = sizeof(cl_uint);
}

/*
 * @inOut: memory object data transfer direction
 */
static __inline void
initMemobjKarg(
    KernelArg *karg,
    cl_mem memobj,
    void *hostBuf,
    size_t hostBufLen,
    MemobjDir dir)
{
    karg->arg.mem = memobj;
    karg->typeSize = sizeof(cl_mem);
    karg->hostBuf = hostBuf;
    karg->hostBufLen = hostBufLen;
    karg->dir = dir;
}

#ifdef __cplusplus
}
#endif

#endif /* CLARGS_H_ */
