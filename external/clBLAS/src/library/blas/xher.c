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

//#define DO_HER

#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

clblasStatus
doher(
	CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
{
	cl_int err;
    ListHead seq;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

	#ifdef DEBUG_HER
	printf("doher called\n");
	#endif

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, X, 0, false, A_MAT_ERRSET, X_VEC_ERRSET, END_ERRSET))) {
   		printf("Invalid mem object..\n");
        return retCode;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, N, N, A, offa, lda, A_MAT_ERRSET))) {
        printf("Invalid Size for A\n");
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, X, offx, incx, X_VEC_ERRSET))) {
        printf("Invalid Size for X\n");
        return retCode;
    }

	if ((commandQueues == NULL) || (numCommandQueues == 0))
    {
        return clblasInvalidValue;
    }

    if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
    {
        return clblasInvalidEventWaitList;
    }

	kargs->order = order;
	if(order == clblasRowMajor)
	{
		kargs->uplo = (uplo == clblasUpper) ? clblasLower : clblasUpper;
	}
	else
	{
		kargs->uplo = uplo;
	}

	kargs->N = N;
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->B = X;
    kargs->ldb.vector = incx;
    kargs->offBX = offx;
    kargs->offa = offa;
	kargs->offA = offa;

	#ifdef DEBUG_HER
    printf("Calling makeSolutionSeq : HER\n");
    #endif

	/*
 	 * Always use commandQueues (0)
	 * PENDING:
	 * 1. No Multi-GPU / Multi-command queue support
	 * 2. This can be optimized to use the commandQ with the higher
	 *	  memmory bandwidth that supports the data-type and the LDA
	 */
	numCommandQueues = 1;

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_HER, kargs, numCommandQueues, commandQueues,
                          numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);
    return (clblasStatus)err;
}


clblasStatus
clblasCher(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
	{
		CLBlasKargs kargs;

    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_COMPLEX_FLOAT;
		kargs.alpha.argFloat = alpha;
        kargs.pigFuncID = CLBLAS_HER;

		#ifdef DEBUG_HER
		printf("CHER called\n");
		#endif

		return doher(&kargs, order, uplo, N, X, offx, incx, A, offa, lda, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZher(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
    {
        CLBlasKargs kargs;

        memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_DOUBLE;
		kargs.alpha.argDouble = alpha;
        kargs.pigFuncID = CLBLAS_HER;

        #ifdef DEBUG_HER
        printf("ZHER called\n");
        #endif

        return doher(&kargs, order, uplo, N, X, offx, incx, A, offa, lda, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clblasChpr(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
	{
		CLBlasKargs kargs;

    	memset(&kargs, 0, sizeof(kargs));
    	kargs.dtype = TYPE_COMPLEX_FLOAT;
		kargs.alpha.argFloat = alpha;
        kargs.pigFuncID = CLBLAS_HPR;

		return doher(&kargs, order, uplo, N, X, offx, incx, AP, offa, 0, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZhpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
    {
        CLBlasKargs kargs;

        memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_DOUBLE;
		kargs.alpha.argDouble = alpha;
        kargs.pigFuncID = CLBLAS_HPR;

        return doher(&kargs, order, uplo, N, X, offx, incx, AP, offa, 0, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

