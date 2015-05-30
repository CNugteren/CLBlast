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


#ifndef SOLUTION_SEQ_H_
#define SOLUTION_SEQ_H_

#include <list.h>
#include <granulation.h>
#include <kern_cache.h>
#include <kernel_extra.h>

#include "blas_funcs.h"
#include "clblas-internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// subproblem dimension components
typedef enum SDimConponent {
    SDIM_X,
    SDIM_Y,
    SDIM_BWIDTH
} SDimComponent;

typedef struct SolutionStep {
    BlasFunctionID funcID;
    Kernel *kernels[MAX_CLBLAS_KERNELS_PER_STEP];
    CLBlasKargs args;
    cl_command_queue cmdQueue;
    TargetDevice device;
    cl_uint numEventsInWaitList;
    const cl_event *eventWaitList;
    cl_event *event;
    unsigned int patternID;
    SubproblemDim subdims[MAX_SUBDIMS];
    PGranularity pgran;
    KernelExtraFlags extraFlags;
    ListNode node;
} SolutionStep;

/**
 * @internal
 * @brief Make solution sequence
 *
 * @param[in] funcID                BLAS function ID
 * @param[in] args                  BLAS parameters
 * @param[in] numCommandQueues      Number of the command queues
 * @param[in] commandQueues         Command queues to distribute the problem
 *                                  among
 * @param[in] numEventsInWaitList   Number of events in the wait list
 * @param[in] eventWaitList         List of events which must fire before any
 *                                  of the problem's kernels can be executed
 * @param[out] events               List of output events signaling on
 *                                  completion of evaluating the problem for
 *                                  the command queues.
 * @param[out] seq                  Solution sequence head which will be
 *                                  followed by all needed solution steps
 *                                  after the function returns
 *
 * @returns
 *     - \b CL_SUCCESS on success;
 *     - \b CL_INVALID_VALUE if \b numCommandQueues is zero, or
 *       \b commandQueues is NULL;
 *     - \b CL_INVALID_DEVICE if the function ID indicates that this is
 *        a double precision function, but any of the command queue's devices
 *        does not support double precision;
 *     - \b CL_INVALID_COMMAND_QUEUE if any of the passed command queues is
 *        invalid;
 *     - \b CL_OUT_OF_HOST_MEMORY if there is not enough memory to allocate
 *        internal structures;
 *     - \b CL_OUT_OF_HOST_RESOURCES if required scratch resources are
 *        unavailable.
 *
 * @ingroup SUBMIT_PROBLEM
 */
cl_int
makeSolutionSeq(
    BlasFunctionID funcID,
    const CLBlasKargs *args,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events,
    ListHead *seq);

/**
 * @internal
 * @brief Free solution sequence
 *
 * @param[out] seq                  Solution sequence to free
 *
 * It initializes the list after freeing.
 *
 * @ingroup SUBMIT_PROBLEM
 */
void
freeSolutionSeq(ListHead *seq);

void
freeSolutionStep(ListNode *node);

/**
 * @internal
 * @brief Execute solution sequence
 *
 * @param[in] seq                   Sequence to execute
 *
 * @returns CL_SUCCESS on success, errors from a clEnqueueNDRangeKernel() call
 *          otherwise.
 *
 * @ingroup SUBMIT_PROBLEM
 */
cl_int
executeSolutionSeq(const ListHead *seq);

/*
 * Get math decomposition of a solution step in order
 * to accelerate its evaluation of faster kernels for
 * other functions. The step must inserted into a
 * solution sequence.
 */
ListNode
*decomposeProblemStep(SolutionStep *step);

cl_int
selectVectorization(const SolutionStep *step, CLBLASKernExtra *kextra);

// Find vector length which lda and tile width is divisible on
unsigned int appropriateVecLen(size_t ld, unsigned int typeSize,
                               size_t tileWidth, int funcLevel);

KernelExtraFlags VISIBILITY_HIDDEN
clblasArgsToKextraFlags(
    const CLBlasKargs *args,
    BlasFunctionID funcID);

void VISIBILITY_HIDDEN
getStepGranulation(SolutionStep *step);

bool VISIBILITY_HIDDEN
dimensionsExceedProblemSize(SolutionStep *step);

void VISIBILITY_HIDDEN
getMinimalStepGranulation(SolutionStep *step);

void VISIBILITY_HIDDEN
detectProblemTails(SolutionStep *step);

void VISIBILITY_HIDDEN
detectOffsets(SolutionStep *step);

unsigned int VISIBILITY_HIDDEN
selectPattern( SolutionStep* pStep, unsigned int maxImages);

void VISIBILITY_HIDDEN
fixupGemmOffsets(CLBlasKargs *kargs, KernelExtraFlags kflags, size_t offsetK);

#ifdef __cplusplus
}
#endif

#endif  /* SOLUTION_SEQ_H_ */
