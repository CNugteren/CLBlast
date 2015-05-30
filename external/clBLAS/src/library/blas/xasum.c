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
#include <stdlib.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

clblasStatus
doAsum(
	CLBlasKargs *kargs,
    size_t N,
    cl_mem asum,
    size_t offAsum,
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
        cl_int err;
		ListHead seq, seq2;
        clblasStatus retCode = clblasSuccess;
        cl_event firstAsumCall;
        CLBlasKargs redctnArgs;
        ListNode *listNodePtr;
        SolutionStep *step;

        DataType asumType = (kargs->dtype == TYPE_COMPLEX_FLOAT) ? TYPE_FLOAT:
                                ((kargs->dtype == TYPE_COMPLEX_DOUBLE) ? TYPE_DOUBLE: kargs->dtype);

		if (!clblasInitialized) {
        return clblasNotInitialized;
		}

		/* Validate arguments */

		retCode = checkMemObjects(scratchBuff, asum, X, true, X_VEC_ERRSET, X_VEC_ERRSET, X_VEC_ERRSET );
		if (retCode) {
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET ))) {
			printf("Invalid Size for X\n");
            return retCode;
		}
		// Minimum size of scratchBuff is N
		if ((retCode = checkVectorSizes(kargs->dtype, N, scratchBuff, 0, 1, X_VEC_ERRSET ))) {
			printf("Insufficient ScratchBuff\n");
            return retCode;
		}

		if ((retCode = checkVectorSizes(asumType, 1, asum, offAsum, 1, X_VEC_ERRSET ))) {
			printf("Invalid Size for asum\n");
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
		kargs->A = asum;
        kargs->offA = offAsum;
		kargs->B = X;
		kargs->offBX = offx;
		kargs->ldb.vector = incx;   // Will be using this as incx
        if(incx <1){
            kargs->N = 1;
        }
        kargs->D = scratchBuff;
        kargs->redctnType = REDUCE_BY_SUM;
        memcpy(&redctnArgs, kargs, sizeof(CLBlasKargs));

        redctnArgs.dtype = asumType;

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_ASUM, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, &firstAsumCall, &seq);
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
                           1, &firstAsumCall, events, &seq2);

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
clblasSasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
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
    CLBlasKargs kargs;
	#ifdef DEBUG_ASUM
	printf("SASUM Called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_ASUM;

    return doAsum(&kargs, N, asum, offAsum, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
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
    CLBlasKargs kargs;
	#ifdef DEBUG_ASUM
	printf("DASUM called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_ASUM;

    return doAsum(&kargs, N, asum, offAsum, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasScasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
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
    CLBlasKargs kargs;
    #ifdef DEBUG_ASUM
    printf("SCASUM Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_ASUM;
    kargs.dtype = TYPE_COMPLEX_FLOAT;

    return doAsum(&kargs, N, asum, offAsum, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDzasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
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
    CLBlasKargs kargs;
    #ifdef DEBUG_DZASUM
    printf("DZASUM Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_ASUM;
    kargs.dtype = TYPE_COMPLEX_DOUBLE;

    return doAsum(&kargs, N, asum, offAsum, X, offx, incx, scratchBuff,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}
