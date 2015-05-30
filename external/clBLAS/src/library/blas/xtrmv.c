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
doTrmv(
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

    if ((retCode = checkMatrixSizes(kargs->dtype, order, trans, N, N, A, offa, lda, A_MAT_ERRSET))) {
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

	#ifdef DEBUG_TRMV
	printf("DoTrmv being called...\n");
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
 	 * doTRMV assumes "commandQueue" of 0. The same is reflected in
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
    kargs->N = N;
    kargs->K = N; //store original N
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->B = x;
    kargs->ldb.vector = incx;
    kargs->C = y;
    kargs->ldc.vector = incx;
    kargs->offBX = offx;
    kargs->offCY = 0; // Not used by assignKargs(); Just for clarity
	kargs->offa = offa;
	kargs->offA = offa;
    kargs->offsetM = 0;
    kargs->offsetN = 0;
//    kargs->offsetK = 0;
    kargs->scimage[0] = 0;
    kargs->scimage[1] = 0;

	#ifdef DEBUG_TRMV
	printf("Calling makeSolutionSeq : TRMV\n");
	#endif

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_TRMV, kargs, numCommandQueues, commandQueues,
        				  numEventsInWaitList+1, newEventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
       	err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);
	free(newEventWaitList);
    return (clblasStatus)err;
}

clblasStatus
clblasStrmv(
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
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
	#ifdef DEBUG_TRMV
	printf("STRMV Called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_TRMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtrmv(
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
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
	#ifdef DEBUG_TRMV
	printf("DTRMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_TRMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}


clblasStatus
clblasCtrmv(
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
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
	#ifdef DEBUG_TRMV
	printf("CTRMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_TRMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtrmv(
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
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;
	#ifdef DEBUG_TRMV
	printf("ZTRMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_TRMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, A, offa, lda, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}


clblasStatus
clblasStpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
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
	#ifdef DEBUG_TPMV
	printf("STPMV Called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_TPMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, AP, offa, 0 /* lda as zero for packed */, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
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
	#ifdef DEBUG_TPMV
	printf("DTPMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_TPMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, AP, offa, 0, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}


clblasStatus
clblasCtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
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
	#ifdef DEBUG_TPMV
	printf("CTPMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_TPMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, AP, offa, 0, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtpmv(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    clblasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
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
	#ifdef DEBUG_TPMV
	printf("ZTPMV called\n");
	#endif

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_TPMV;

    return doTrmv(&kargs, order, uplo, trans, diag, N, AP, offa, 0, X, offx, incx, scratchBuff, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}
