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

//#define DEBUG_TRSV

static clblasUplo
getUpLo(CLBlasKargs *kargs)
{
	if (kargs->order == clblasColumnMajor)
	{
		return kargs->uplo;
	}

	if (kargs->uplo == clblasUpper)
	{
		return clblasLower;
	}
	return clblasUpper;
}


static clblasStatus
orchestrateNonTransposeTRSV(CLBlasKargs *kargs, ListHead *trtriSeq, ListHead *gemvSeq, cl_uint numEventsInWaitList,
				const cl_event *eventWaitList, cl_event *events)
{
	clblasStatus err;
	SolutionStep *trtri, *gemv;
	size_t nLoops, i;
	cl_event *eventArray;
	size_t TARGET_ROWS;

	ListNode *f = listNodeFirst(trtriSeq);
	trtri = container_of(f, node, SolutionStep);
	f = listNodeFirst(gemvSeq);
	gemv = container_of(f, node, SolutionStep);
	TARGET_ROWS = trtri->subdims->y;

	if ((trtri->subdims->y) != (gemv->subdims->y))
	{
		printf("TRSV: WARNING:	TRTRI and GEMV dont have identical sub-divisions!!! %lu and %lu\n", trtri->subdims->y, gemv->subdims->y);
		return clblasNotImplemented;
	} else {
		#ifdef DEBUG_TRSV
		printf("TRSV: MESSAGE:	TRTRI and GEMV have identical sub-divisions! = %lu\n", TARGET_ROWS);
		#endif
	}

	trtri->numEventsInWaitList = numEventsInWaitList;
	trtri->eventWaitList = eventWaitList;

	if (kargs->N <= TARGET_ROWS)
	{
		trtri->event = events;
		trtri->args.startRow = 0;
		trtri->args.endRow = (cl_int)((kargs->N)-1);
		err = executeSolutionSeq(trtriSeq);
		return err;
	}

	//
	// Allocate Event Chain
	//
	nLoops = ((kargs->N) / TARGET_ROWS);
	if ((kargs->N % TARGET_ROWS))
	{
		nLoops++;
	}
	#ifdef DEBUG_TRSV
	printf("TRSV: Orchestrate No Transpose Case: nLoops = %d\n", nLoops);
	#endif
	eventArray = malloc(nLoops*sizeof(cl_event));
	if (eventArray == NULL)
	{
		return clblasOutOfHostMemory;
	}

	//
	//	Solve 1 Triangle using Triangle Kernel Followed by Rectangle Kernels
	//
	trtri->event = &eventArray[0];
	if (getUpLo(kargs) == clblasUpper)
	{
		trtri->args.startRow = (cl_int)((kargs->N) - TARGET_ROWS);
		trtri->args.endRow = (cl_int)((kargs->N)-1);
	} else {
		trtri->args.startRow = 0;
		trtri->args.endRow = (cl_int)(TARGET_ROWS-1);
	}
	err = executeSolutionSeq(trtriSeq);
	if (err == CL_SUCCESS)
	{
		//
		// Solve the Rectangles one by one
		//
		for(i=1; i<nLoops; i++)
		{
			gemv->numEventsInWaitList = 1;
			gemv->eventWaitList = &eventArray[i-1];
			if (i < (nLoops-1))
			{
				gemv->event = &eventArray[i];
			} else {
				gemv->event = events;
			}

			if (getUpLo(kargs) == clblasUpper)
			{
				gemv->args.startRow = (cl_int)((kargs->N-1) - (i-1)*TARGET_ROWS);
				gemv->args.endRow   = (cl_int)((kargs->N) - (i)*TARGET_ROWS);
			} else {
				gemv->args.startRow = (cl_int)((i-1)*TARGET_ROWS);
				gemv->args.endRow   = (cl_int)((kargs->N) - (TARGET_ROWS*i));
			}
			err = executeSolutionSeq(gemvSeq);
			if (err != CL_SUCCESS)
			{
				printf("TRSV: WARNING: GEMV LOOP: Breaking after %d iterations	!!!\n", (int)i);
				break;
			}
		}
	}

	free(eventArray);
	return err;
}

static clblasStatus
orchestrateTransposeTRSV(CLBlasKargs *kargs, ListHead *trtriSeq, ListHead *gemvSeq, cl_uint numEventsInWaitList,
				const cl_event *eventWaitList, cl_event *events)
{
	clblasStatus err;
	SolutionStep *trtri, *gemv;
	size_t nLoops, i;
	cl_event *triangleEventArray;
	cl_event *rectangleEventArray;
	size_t TRIANGLE_HEIGHT;

	ListNode *f = listNodeFirst(trtriSeq);
	trtri = container_of(f, node, SolutionStep);
	f = listNodeFirst(gemvSeq);
	gemv = container_of(f, node, SolutionStep);
	TRIANGLE_HEIGHT = trtri->subdims->y;

	if ((trtri->subdims->y) != (gemv->subdims->y))
	{
		printf("TRSV: Transpose: WARNING: TRTRI and GEMV dont have identical sub-divisions!!! %lu and %lu\n", trtri->subdims->y, gemv->subdims->y);
		return clblasNotImplemented;
	} else {
		#ifdef DEBUG_TRSV
		printf("TRSV: Transpose: MESSAGE:	TRTRI and GEMV have identical sub-divisions! = %lu\n", TRIANGLE_HEIGHT);
		#endif
	}

	trtri->numEventsInWaitList = numEventsInWaitList;
	trtri->eventWaitList = eventWaitList;
	if (kargs->N <= TRIANGLE_HEIGHT)
	{
		trtri->event = events;
		trtri->args.startRow = 0;
		trtri->args.endRow = (cl_int)(kargs->N);
		err = executeSolutionSeq(trtriSeq);
		return err;
	}

	//
	// Allocate Event Chain
	//
	nLoops = ((kargs->N) / TRIANGLE_HEIGHT);
	if ((kargs->N % TRIANGLE_HEIGHT))
	{
		nLoops++;
	}
	#ifdef DEBUG_TRSV
	printf("nLoops: %d\n", nLoops);
	#endif
	//
	// Allocate Event Arrays to order the orchestration
	//
	triangleEventArray = malloc(nLoops*sizeof(cl_event));
	rectangleEventArray = malloc(nLoops*sizeof(cl_event));
	if ((triangleEventArray == NULL) || (rectangleEventArray == NULL))
	{
		if (triangleEventArray)
		{
			free (triangleEventArray);
		}
		if (rectangleEventArray)
		{
			free (rectangleEventArray);
		}
		return clblasOutOfHostMemory;
	}

	//
	//	Solve as chain of TRIANGLE, RECTANGLE kernels ending on a pair-less TRIANGLE
	//
	for(i=0; i<nLoops; i++)
	{
		//
		// TRIANGLE EXECUTION
		//
		#ifdef DEBUG_TRSV
		printf("Calling TRTRI-");
		#endif
		trtri->event = &triangleEventArray[i];
		if (i == (nLoops-1))
		{
			//
			// TRTRI's last iteration must be tied to the "event" that the API
			// user will choose to wait on.
			//
			trtri->event = events;
		}

		if (i != 0)
		{
			//
			// For first iteration, TRTRI waits on what the API user has specified.
			// Subsequent iterations will wait on the previous iteration's rectangle
			// counterpart
			//
			trtri->numEventsInWaitList =1;
			trtri->eventWaitList = &rectangleEventArray[i-1];
		}

		if (getUpLo(kargs) == clblasUpper)
		{
			trtri->args.startRow 	= (cl_int)(TRIANGLE_HEIGHT*i);
			trtri->args.endRow 		= (cl_int)(TRIANGLE_HEIGHT*(i+1));
			if (trtri->args.endRow >= (cl_int)kargs->N)
			{
				trtri->args.endRow = (cl_int)kargs->N;
			}
		} else {
			if (kargs->N < TRIANGLE_HEIGHT*(i+1))
			{
				trtri->args.startRow 	= 0;
			} else {
				trtri->args.startRow 	= (cl_int)((kargs->N) - TRIANGLE_HEIGHT*(i+1));
			}
			trtri->args.endRow 		= (cl_int)((kargs->N) - TRIANGLE_HEIGHT*(i));
		}
		err = executeSolutionSeq(trtriSeq);
		if (err != CL_SUCCESS)
		{
			printf("TRSV: Transpose: Breaking in the middle of loop due to error status, i=%d\n", (int)i);
			break;
		}
		if (i == (nLoops-1))
		{
			break;
		}
		#ifdef DEBUG_TRSV
		printf("Calling gemv-");
		#endif
		gemv->numEventsInWaitList = 1;
		gemv->eventWaitList = &triangleEventArray[i];
		gemv->event = &rectangleEventArray[i];
		gemv->args.startRow = trtri->args.startRow;
		gemv->args.endRow = trtri->args.endRow;
		err = executeSolutionSeq(gemvSeq);
		if (err != CL_SUCCESS)
		{
			printf("TRSV: Transpose: WARNING: GEMV LOOP: Breaking after %d iterations	!!!\n", (int)i);
			break;
		}
	}

	free(triangleEventArray);
	free(rectangleEventArray);
	return err;
}

static clblasStatus
orchestrateTRSV(CLBlasKargs *kargs, ListHead *trtriSeq, ListHead *gemvSeq, cl_uint numEventsInWaitList,
				const cl_event *eventWaitList, cl_event *events)
{
	clblasStatus err = clblasNotImplemented;

	if 	(	((kargs->order == clblasColumnMajor) && (kargs->transA == clblasNoTrans))	||
			((kargs->order == clblasRowMajor) && (kargs->transA != clblasNoTrans))
		)
	{
		#ifdef DEBUG_TRSV
		printf("Orchestrating the NO-Transpose case..\n");
		#endif
		err = orchestrateNonTransposeTRSV(kargs, trtriSeq, gemvSeq, numEventsInWaitList, eventWaitList, events);
	} else {
		#ifdef DEBUG_TRSV
		printf("Orchestrating the Transpose case..\n");
		#endif
		err = orchestrateTransposeTRSV(kargs, trtriSeq, gemvSeq, numEventsInWaitList, eventWaitList, events);
	}

	return err;
}

clblasStatus
doTrsv(
	CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem x,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err = clblasNotImplemented;
    ListHead seq;
	CLBlasKargs gemvKargs;
	ListHead gemvSeq;
	// cl_context c;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, x, (cl_mem) NULL, false, A_MAT_ERRSET, X_VEC_ERRSET, END_ERRSET))) {
		#ifdef DEBUG_TRSV
		printf("Invalid mem object..\n");
		#endif
        return retCode;
    }

	/*
	 * PENDING:
 	 * checkMatrixSizes() does not account for "offa" argument.
 	 * Need to pass "offa" when "checkMatrixSizes()" is changed.
	 */
    if ((retCode = checkMatrixSizes(kargs->dtype, order, trans, N, N, A, offa, lda, A_MAT_ERRSET))) {
		#ifdef DEBUG_TRSV
		printf("Invalid Size for A\n");
		#endif
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, x, offx, incx, X_VEC_ERRSET))) {
		#ifdef DEBUG_TRSV
		printf("Invalid Size for X\n");
		#endif
        return retCode;
    }

	#ifdef DEBUG_TRSV
	printf("DoTrsv being called...\n");
	#endif

	if ((commandQueues == NULL) || (numCommandQueues == 0))
	{
		return clblasInvalidValue;
	}

	if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
	{
		return clblasInvalidEventWaitList;
	}

    if (commandQueues[0] == NULL)
	{
		return clblasInvalidCommandQueue;
	}

	numCommandQueues = 1; // NOTE: Hard-coding the number of command queues to 1
    kargs->order = order;
    kargs->uplo = uplo;
    kargs->transA = trans;
	kargs->diag = diag;
    kargs->M = N; // store Original N
    kargs->N = N; // The field "kargs->N" is the one used by the generator.
    kargs->K = N; // store original N
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->B = x;
    kargs->ldb.vector = incx;
    kargs->offBX = offx;
	kargs->offa = offa;
	kargs->offA = offa;
    kargs->offsetM = 0;
    kargs->offsetN = 0;
    kargs->scimage[0] = 0;
    kargs->scimage[1] = 0;
	memcpy(&gemvKargs, kargs, sizeof(CLBlasKargs));

	#ifdef DEBUG_TRSV
	printf("Calling makeSolutionSeq : TRSV\n");
	#endif

    listInitHead(&seq);
	listInitHead(&gemvSeq);
    //err = makeSolutionSeq(CLBLAS_TRSV, kargs, numCommandQueues, commandQueues,
        				  //0, NULL, NULL, &seq);

	/*
    Problem of context getting released on entry seems to be gone on the new driver.
    Uncomment these lines if problem recurs

    getQueueContext(commandQueues[0], &c);
	clRetainContext(c);
	#ifdef DEBUG_TRSV
	clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &refcnt, NULL);
	printf("doTrsv(): REFCNT ON ENTRY= %u\n", refcnt);
	#endif
    */

    err = makeSolutionSeq(CLBLAS_TRSV, kargs, numCommandQueues, commandQueues,
        				  numEventsInWaitList, eventWaitList, events, &seq);
	if (err == CL_SUCCESS)
	{
		err = makeSolutionSeq(CLBLAS_TRSV_GEMV, &gemvKargs, numCommandQueues, commandQueues,
								0, NULL, NULL, &gemvSeq);
		if (err == CL_SUCCESS)
		{
			#ifdef DEBUG_TRSV
			printf("Orchestrating TRSV\n");
			#endif
			err = orchestrateTRSV(kargs, &seq, &gemvSeq, numEventsInWaitList, eventWaitList, events);
		}
	}

    freeSolutionSeq(&seq);
	freeSolutionSeq(&gemvSeq);
	#ifdef DEBUG_TRSV
	if (clGetContextInfo(c, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &refcnt, NULL) != CL_SUCCESS)
	{
		printf("doTrsv(): clGetContextInfo failed..\n");
	} else {
		printf("doTrsv(): REFCNT EXIT = %u\n", refcnt);
	}
	#endif
    return  err;
}

clblasStatus
clblasStrsv(
clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("STRSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_TRSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("DTRSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_TRSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("CTRSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_TRSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtrsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("ZTRSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_TRSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasStpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("STPSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_TPSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, 0, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("DTPSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_TPSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, 0, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("CTPSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_TPSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, 0, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtpsv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
    #ifdef DEBUG_TRSV
    printf("ZTPSV Called\n");
    #endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_TPSV;

    return doTrsv(&kargs, order, uplo, trans, diag, N, A, offa, 0, X, offx, incx, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

