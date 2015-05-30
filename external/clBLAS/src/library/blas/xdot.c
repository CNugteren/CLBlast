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
doDot(
	CLBlasKargs *kargs,
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    int doConj,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
        cl_int err;
		ListHead seq, seq2;
        clblasStatus retCode = clblasSuccess;
        cl_event firstDotCall;
        CLBlasKargs redctnArgs;
        ListNode *listNodePtr;
        SolutionStep *step;

		if (!clblasInitialized) {
        return clblasNotInitialized;
		}

		/* Validate arguments */

		retCode = checkMemObjects(X, Y, X, false, X_VEC_ERRSET, Y_VEC_ERRSET, X_VEC_ERRSET );
		retCode |= checkMemObjects(scratchBuff, dotProduct, X, false, X_VEC_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET );
		if (retCode) {
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET))) {
			printf("Invalid Size for X\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, N, Y, offy, incy, Y_VEC_ERRSET))) {
			printf("Invalid Size for Y\n");
            return retCode;
		}
		// Minimum size of scratchBuff is N
		if ((retCode = checkVectorSizes(kargs->dtype, N, scratchBuff, 0, 1, X_VEC_ERRSET))) {
			printf("Insufficient ScratchBuff\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, 1, dotProduct, offDP, 1, Y_VEC_ERRSET))) {
			printf("Invalid Size for dotProduct\n");
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
		kargs->A = dotProduct;
        kargs->offA = offDP;
        kargs->offa = offDP;
		kargs->B = X;
		kargs->offBX = offx;
		kargs->ldb.vector = incx;   // Will be using this as incx
		kargs->C = Y;
		kargs->offCY = offy;
		kargs->ldc.vector = incy;	// Will be using this as incy
        kargs->D = scratchBuff;
        kargs->redctnType = REDUCE_BY_SUM;
        kargs->K = (size_t)doConj;
        memcpy(&redctnArgs, kargs, sizeof(CLBlasKargs));

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_DOT, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, &firstDotCall, &seq);
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
                           1, &firstDotCall, events, &seq2);

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
clblasSdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    int doConj;
	#ifdef DEBUG_DOT
	printf("SDOT Called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_DOT;
    doConj = 0;

    return doDot(&kargs, N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, doConj,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDdot(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    int doConj;
	#ifdef DEBUG_DOT
	printf("DDOT called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_DOT;
    doConj = 0;

    return doDot(&kargs, N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, doConj,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCdotu(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    int doConj;
    #ifdef DEBUG_DOT
    printf("CDOTU Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_DOT;
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    doConj = 0;

    return doDot(&kargs, N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, doConj,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZdotu(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    int doConj;
    #ifdef DEBUG_DOT
    printf("ZDOTU Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_DOT;
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    doConj = 0;

    return doDot(&kargs, N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, doConj,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCdotc(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    int doConj;
    #ifdef DEBUG_DOT
    printf("CDOTU Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_DOT;
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    doConj = 1;

    return doDot(&kargs, N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, doConj,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZdotc(
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    int doConj;
    #ifdef DEBUG_DOT
    printf("ZDOTU Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.pigFuncID = CLBLAS_DOT;
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    doConj = 1;

    return doDot(&kargs, N, dotProduct, offDP, X, offx, incx, Y, offy, incy, scratchBuff, doConj,
                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
}
