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

//#define DEBUG_TBMV

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

clblasStatus
doTbmv(
	CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem x,
    size_t offx,
    int incx,
	cl_mem y, // Scratch Buffer
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err;
    ListHead seq;
	size_t sizeOfVector;
	cl_event *newEventWaitList;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, x, y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET))) {
	printf("Invalid mem object..\n");
        return retCode;
    }

    if ((retCode = checkBandedMatrixSizes(kargs->dtype, order, trans, N, N, K, 0, A, offa, lda, A_MAT_ERRSET))) {
		printf("Invalid Size for A\n");
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, x, offx, incx, X_VEC_ERRSET))) {
		printf("Invalid Size for X\n");
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, y, 0, incx, Y_VEC_ERRSET))) {
		printf("Invalid Size for scratch vector\n");
        return retCode;
    }

	#ifdef DEBUG_TBMV
	printf("DoTbmv being called...\n");
	#endif

	if ((commandQueues == NULL) || (numCommandQueues == 0))
	{
		return clblasInvalidValue;
	}
    numCommandQueues = 1;

	if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
	{
		return clblasInvalidEventWaitList;
	}

	newEventWaitList = malloc((numEventsInWaitList+1) * sizeof(cl_event));
	if (newEventWaitList == NULL)
	{
		return clblasOutOfHostMemory;
	}
	if (numEventsInWaitList != 0 )
	{
		memcpy(newEventWaitList, eventWaitList, numEventsInWaitList*sizeof(cl_event));
	}

	/*
 	 * ASSUMPTION:
 	 * doTBMV assumes "commandQueue" of 0. The same is reflected in
	 * "makeSolutionSeq" as well. If either of them changes in future,
	 * this code needs to be revisited.
  	 */
	sizeOfVector = (1 + (N-1)*abs(incx)) * dtypeSize(kargs->dtype);
	err = clEnqueueCopyBuffer(commandQueues[0], x, y, offx*dtypeSize(kargs->dtype), 0, sizeOfVector,
							  numEventsInWaitList, eventWaitList, &newEventWaitList[numEventsInWaitList]);
	if (err != CL_SUCCESS)
	{
		free(newEventWaitList);
		return err;
	}

    kargs->order = order;
    kargs->uplo = uplo;
    kargs->transA = trans;
	kargs->diag = diag;
	kargs->M = N;
    kargs->N = N;
    if( uplo == clblasUpper )
    {
        kargs->KL = 0;
        kargs->KU = K;
    }
    else    {
        kargs->KL = K;
        kargs->KU = 0;
    }
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->B = y;       // Now it becomes x = A * y
    kargs->ldb.vector = incx;
    kargs->C = x;
    kargs->ldc.vector = incx;
    kargs->offBX = 0;           // Not used by assignKargs(); Just for clarity
    kargs->offCY = offx;
	kargs->offa = offa;
	kargs->offA = offa;

	#ifdef DEBUG_TBMV
	printf("Calling makeSolutionSeq : TBMV\n");
	#endif

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_GBMV, kargs, numCommandQueues, commandQueues,
        				  numEventsInWaitList+1, newEventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
       	err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);
	free(newEventWaitList);
    return (clblasStatus)err;
}

clblasStatus
clblasStbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
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
	#ifdef DEBUG_TBMV
	printf("STBMV Called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_TBMV;

    return doTbmv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
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
	#ifdef DEBUG_TBMV
	printf("DTBMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_TBMV;

    return doTbmv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}


clblasStatus
clblasCtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
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
	#ifdef DEBUG_TBMV
	printf("CTBMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_TBMV;

    return doTbmv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtbmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
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
	#ifdef DEBUG_TBMV
	printf("ZTBMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_TBMV;

    return doTbmv(&kargs, order, uplo, trans, diag, N, K, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}
