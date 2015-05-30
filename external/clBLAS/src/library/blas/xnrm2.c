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

//#define USE_HYPOT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

clblasStatus
doNrm2_hypot(CLBlasKargs *kargs,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err;
	ListHead seq, seq2;
    cl_event firstNrmCall;
    CLBlasKargs redctnArgs;
    ListNode *listNodePtr;
    SolutionStep *step;

    //
    // Scratch buffer will be of %PTYPE
    // Result of compelx nrm2 is scalar
    //
    DataType nrmType = (kargs->dtype == TYPE_COMPLEX_FLOAT)? TYPE_FLOAT :
                       ((kargs->dtype == TYPE_COMPLEX_DOUBLE)? TYPE_DOUBLE : (kargs->dtype));

    kargs->redctnType = REDUCE_BY_HYPOT;
    memcpy(&redctnArgs, kargs, sizeof(CLBlasKargs));
    redctnArgs.dtype = nrmType;

	listInitHead(&seq);
	err = makeSolutionSeq(CLBLAS_NRM2, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, &firstNrmCall, &seq);
	if (err == CL_SUCCESS)
    {
        /** The second kernel call needs to know the number of work-groups used
            in the first kernel call. This number of work-groups is calculated here
            and passed as N to second reduction kernel
        **/
        err = executeSolutionSeq(&seq);
        if (err == CL_SUCCESS)
        {
            listNodePtr = listNodeFirst(&seq);        // Get the node
            step = container_of(listNodePtr, node, SolutionStep);
            redctnArgs.N = step->pgran.numWGSpawned[0];     // 1D block was used

            listInitHead(&seq2);
            err = makeSolutionSeq(CLBLAS_REDUCTION_EPILOGUE, &redctnArgs, numCommandQueues, commandQueues,
                      1, &firstNrmCall, events, &seq2);

            if (err == CL_SUCCESS)
            {
                err = executeSolutionSeq(&seq2);
            }
            freeSolutionSeq(&seq2);
        }
    }

	freeSolutionSeq(&seq);
	return (clblasStatus)err;
}

clblasStatus
doNrm2_ssq(CLBlasKargs *kargs,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err;
	ListHead seq, seq2;
    cl_event firstNrmCall;
    CLBlasKargs redctnArgs;
    ListNode *listNodePtr;
    SolutionStep *step;

    //
    // Scratch buffer will be of %PTYPE
    // Result of compelx nrm2 is scalar
    //
    DataType nrmType = (kargs->dtype == TYPE_COMPLEX_FLOAT)? TYPE_FLOAT :
                       ((kargs->dtype == TYPE_COMPLEX_DOUBLE)? TYPE_DOUBLE : (kargs->dtype));

    kargs->redctnType = REDUCE_BY_SSQ;
    memcpy(&redctnArgs, kargs, sizeof(CLBlasKargs));
    redctnArgs.dtype = nrmType;

	listInitHead(&seq);
	err = makeSolutionSeq(CLBLAS_NRM2, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, &firstNrmCall, &seq);
	if (err == CL_SUCCESS)
    {
        /** The second kernel call needs to know the number of work-groups used
            in the first kernel call. This number of work-groups is calculated here
            and passed as N to second reduction kernel
        **/
        err = executeSolutionSeq(&seq);
        if (err == CL_SUCCESS)
        {
            listNodePtr = listNodeFirst(&seq);        // Get the node
            step = container_of(listNodePtr, node, SolutionStep);
            redctnArgs.N = step->pgran.numWGSpawned[0];     // 1D block was used

            listInitHead(&seq2);
            err = makeSolutionSeq(CLBLAS_REDUCTION_EPILOGUE, &redctnArgs, numCommandQueues, commandQueues,
                      1, &firstNrmCall, events, &seq2);

            if (err == CL_SUCCESS)
            {
                err = executeSolutionSeq(&seq2);
            }
            freeSolutionSeq(&seq2);
        }
    }

	freeSolutionSeq(&seq);
	return (clblasStatus)err;
}


clblasStatus
doNrm2(
    bool useHypot,
	CLBlasKargs *kargs,
	size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus retCode = clblasSuccess;

    DataType nrmType = (kargs->dtype == TYPE_COMPLEX_FLOAT)? TYPE_FLOAT :
                       ((kargs->dtype == TYPE_COMPLEX_DOUBLE)? TYPE_DOUBLE : (kargs->dtype));

	if (!clblasInitialized) {
        return clblasNotInitialized;
	}

	/* Validate arguments */

	retCode = checkMemObjects(X, NRM2, scratchBuff, true, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
	if (retCode) {
		printf("Invalid mem object..\n");
        return retCode;
	}

	// Check wheather enough memory was allocated
    retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET );
	if (retCode) {
		printf("Invalid Size for X\n");
        return retCode;
	}
	// Minimum size of scratchBuff is 2*N
	retCode = checkVectorSizes(kargs->dtype, (2*N), scratchBuff, 0, 1, X_VEC_ERRSET );
    if (retCode) {
		printf("Insufficient ScratchBuff\n");
        return retCode;
	}

    retCode = checkVectorSizes(nrmType, 1, NRM2, offNRM2, 1, Y_VEC_ERRSET );
	if (retCode) {
		printf("Invalid Size for NRM2\n");
        return retCode;
	}
	///////////////////////////////////////////////////////////////

	if ((commandQueues == NULL) || (numCommandQueues == 0))
	{
		return clblasInvalidValue;
	}

	/* numCommandQueues will be hardcoded to 1 as of now. No multi-gpu support */
	numCommandQueues = 1;
	if (commandQueues[0] == NULL)
	{
		return clblasInvalidCommandQueue;
	}

	if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
	{
		return clblasInvalidEventWaitList;
	}

	kargs->N = N;
	kargs->A = NRM2;
    kargs->offA = offNRM2;
    kargs->offa = offNRM2;
	kargs->B = X;
	kargs->offBX = offx;
	kargs->ldb.vector = incx;
    if(incx < 1) {              // According to netlib, if incx<1, NRM2 will be zero
        kargs->N = 1;           // Makeing it launch only 1 work-group
    }
    kargs->D = scratchBuff;

    if(useHypot)
    {
        return doNrm2_hypot(kargs, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }
    else
    {
        return doNrm2_ssq(kargs, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }
}

clblasStatus
clblasSnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    bool useHypot;
    CLBlasKargs kargs;

    #ifdef USE_HYPOT
        useHypot = true;
    #else
        useHypot = false;
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;

    return doNrm2(useHypot, &kargs, N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    bool useHypot;
    CLBlasKargs kargs;

    #ifdef USE_HYPOT
        useHypot = true;
    #else
        useHypot = false;
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;

    return doNrm2(useHypot, &kargs, N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasScnrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    bool useHypot;
    CLBlasKargs kargs;

    #ifdef USE_HYPOT
        useHypot = true;
    #else
        useHypot = false;
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;

    return doNrm2(useHypot, &kargs, N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDznrm2(
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    bool useHypot;
    CLBlasKargs kargs;

    #ifdef USE_HYPOT
        useHypot = true;
    #else
        useHypot = false;
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;

    return doNrm2(useHypot, &kargs, N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

