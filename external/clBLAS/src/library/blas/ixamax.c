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

//#define IAMAX_USE_ATOMIC_MIN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

clblasStatus
doiAmax(
	CLBlasKargs *kargs,
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuf,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
        cl_int err;
		ListHead seq, seq2;
        clblasStatus retCode = clblasSuccess;
        cl_event firstiAmaxCall;
        CLBlasKargs redctnArgs;
        ListNode *listNodePtr;
        SolutionStep *step;

		if (!clblasInitialized) {
        return clblasNotInitialized;
		}

		/* Validate arguments */

		retCode = checkMemObjects(X, scratchBuf, iMax, true, X_VEC_ERRSET, A_MAT_ERRSET, X_VEC_ERRSET );
		if (retCode) {
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET ))) {
			printf("Invalid Size for X\n");
            return retCode;
		}
		// Minimum size of scratchBuff is 2 * N
		if ((retCode = checkVectorSizes(kargs->dtype, (2 * N), scratchBuf, 0, 1, A_MAT_ERRSET ))) {
			printf("Insufficient ScratchBuff A\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(TYPE_UNSIGNED_INT, 1, iMax, offiMax, 1, X_VEC_ERRSET ))) {
			printf("Invalid Size for iX\n");
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

        // cl_mem D is scratch buffer
        // cl_mem A is the output Buffer i.e. iMAX, offA for offiMax
        // cl_mem B is the input Buffer containing N Values
		kargs->N = N;
		kargs->B = X;
        kargs->offb = offx;
		kargs->ldb.vector = incx;   // Will be using this as incx
        if(incx < 1) {              // According to netlib, if incx<1, NRM2 will be zero
            kargs->N = 1;           // Makeing it launch only 1 work-group
        }
		kargs->D = scratchBuf;
		kargs->A = iMax;
		kargs->offA = offiMax;
#ifdef IAMAX_USE_ATOMIC_MIN
        kargs->redctnType = REDUCE_MAX_WITH_INDEX_ATOMICS;
#else
        kargs->redctnType = REDUCE_MAX_WITH_INDEX;
#endif
        memcpy(&redctnArgs, kargs, sizeof(CLBlasKargs));

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_iAMAX, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, &firstiAmaxCall, &seq);
		if (err == CL_SUCCESS)
        {
            // The second kernel call needs to know the number of work-groups used
            //    in the first kernel call. This number of work-groups is calculated here
            //    and passed as N to second reduction kernel

            err = executeSolutionSeq(&seq);
            if (err == CL_SUCCESS)
            {
                listNodePtr = listNodeFirst(&seq);        // Get the node
                step = container_of(listNodePtr, node, SolutionStep);

                redctnArgs.N = step->pgran.numWGSpawned[0];     // 1D block was used
                redctnArgs.dtype = (redctnArgs.dtype == TYPE_COMPLEX_FLOAT) ? TYPE_FLOAT :
                    ((redctnArgs.dtype == TYPE_COMPLEX_DOUBLE) ? TYPE_DOUBLE : redctnArgs.dtype);

                listInitHead(&seq2);
                err = makeSolutionSeq(CLBLAS_REDUCTION_EPILOGUE, &redctnArgs, numCommandQueues, commandQueues,
                           1, &firstiAmaxCall, events, &seq2);

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
clblasiSamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuf,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
	#ifdef DEBUG_iAMAX
	printf("iSAMAX Called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_iAMAX;

    return doiAmax(&kargs, N, iMax, offiMax, X, offx, incx, scratchBuf,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasiDamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuf,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
	#ifdef DEBUG_iAMAX
	printf("iDAMAX called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_iAMAX;

    return doiAmax(&kargs, N, iMax, offiMax, X, offx, incx, scratchBuf,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasiCamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuf,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_iAMAX
    printf("iCAMAX Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_iAMAX;
    kargs.dtype = TYPE_COMPLEX_FLOAT;

    return doiAmax(&kargs, N, iMax, offiMax, X, offx, incx, scratchBuf,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasiZamax(
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuf,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_iAMAX
    printf("iZAMAX Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_iAMAX;
    kargs.dtype = TYPE_COMPLEX_DOUBLE;

    return doiAmax(&kargs, N, iMax, offiMax, X, offx, incx, scratchBuf,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

