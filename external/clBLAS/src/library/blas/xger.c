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


//#define DEBUG_GER

#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"


clblasStatus
doGer(
	CLBlasKargs *kargs,
	clblasOrder order,
    size_t M,
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem  A,
    size_t offa,
    size_t lda,
	int doConj,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		cl_int err;
		ListHead seq;
        clblasStatus retCode = clblasSuccess;

		if (!clblasInitialized) {
        return clblasNotInitialized;
		}

		/* Validate arguments */

		if ((retCode = checkMemObjects(A, X, Y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET))) {
			printf("Invalid mem object..\n");
            return retCode;
		}

		// Check wheather enough memory was allocated

		if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, M, N, A, offa, lda, A_MAT_ERRSET))) {

			printf("Invalid Size for A %d\n",retCode );
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, M, X, offx, incx, X_VEC_ERRSET))) {
			printf("Invalid Size for X\n");
            return retCode;
		}
		if ((retCode = checkVectorSizes(kargs->dtype, N, Y, offy, incy, Y_VEC_ERRSET))) {
			printf("Invalid Size for Y\n");
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

		/*
 		 * ASSUMPTION:
 		 * doTRMV assumes "commandQueue" of 0. The same is reflected in
		 * "makeSolutionSeq" as well. If either of them changes in future,
		 * this code needs to be revisited.
  		 */

		kargs->order = order;
		kargs->M = M;
		kargs->N = N;
		kargs->A = A;
		kargs->offa = offa;
		kargs->offA = offa;
		kargs->lda.matrix = lda;
		kargs->B = X;
		kargs->offBX = offx;
		kargs->ldb.vector = incx;	// Will be using this as incx
		kargs->C = Y;
		kargs->offCY = offy;
		kargs->ldc.vector = incy;	// Will be using this as incy
		kargs->offsetM = 0;
		kargs->offsetN = 0;
		kargs->scimage[0] = 0;
		kargs->scimage[1] = 0;
		kargs->K = (size_t)doConj; // Will be using K as doConj parameter

		#ifdef DEBUG_GER
		printf("Calling makeSolutionSeq from DoGer: GER\n");
		#endif

		listInitHead(&seq);
		err = makeSolutionSeq(CLBLAS_GER, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, events, &seq);
		if (err == CL_SUCCESS) {
       		err = executeSolutionSeq(&seq);
		}

		freeSolutionSeq(&seq);

		return (clblasStatus)err;
	}





clblasStatus
clblasSger(
    clblasOrder order,
    size_t M,
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem  A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;
		int doConj;

		#ifdef DEBUG_GER
		printf("\nSGER Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_FLOAT;
		kargs.alpha.argFloat = alpha;
		doConj = 0;

		return doGer(&kargs, order, M, N, X, offx, incx, Y, offy, incy, A, offa, lda, doConj,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasDger(
    clblasOrder order,
    size_t M,
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;
		int doConj;

		#ifdef DEBUG_GER
		printf("\nDGER Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_DOUBLE;
		kargs.alpha.argDouble = alpha;
		doConj = 0;

		return doGer(&kargs, order, M, N, X, offx, incx, Y, offy, incy, A, offa, lda, doConj,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasCgeru(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int  incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;
		int doConj;

		#ifdef DEBUG_GER
		printf("\nCGERU Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_FLOAT;
		kargs.alpha.argFloatComplex = alpha;
		doConj = 0;

		return doGer(&kargs, order, M, N, X, offx, incx, Y, offy, incy, A, offa, lda, doConj,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZgeru(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;
		int doConj;

		#ifdef DEBUG_GER
		printf("\nZGERU Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_DOUBLE;
		kargs.alpha.argDoubleComplex = alpha;
		doConj = 0;

		return doGer(&kargs, order, M, N, X, offx, incx, Y, offy, incy, A, offa, lda, doConj,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasCgerc(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int  incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;
		int doConj;

		#ifdef DEBUG_GER
		printf("\nCGERC Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_FLOAT;
		kargs.alpha.argFloatComplex = alpha;
		doConj = 1;

		return doGer(&kargs, order, M, N, X, offx, incx, Y, offy, incy, A, offa, lda, doConj,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZgerc(
    clblasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		CLBlasKargs kargs;
		int doConj;

		#ifdef DEBUG_GER
		printf("\nZGERC Called\n");
		#endif

		memset(&kargs, 0, sizeof(kargs));
		kargs.dtype = TYPE_COMPLEX_DOUBLE;
		kargs.alpha.argDoubleComplex = alpha;
		doConj = 1;

		return doGer(&kargs, order, M, N, X, offx, incx, Y, offy, incy, A, offa, lda, doConj,
						numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}


