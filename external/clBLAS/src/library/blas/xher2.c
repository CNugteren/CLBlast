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

//#define DEBUG_HER2

#include <stdio.h>
#include <string.h>
#include <clBLAS.h>

#include <devinfo.h>
#include "clblas-internal.h"
#include "solution_seq.h"

clblasStatus
doHer2(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
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
    cl_command_queue* commandQueue,
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

    #ifdef DEBUG_HER2
    printf("doHer2 called\n");
    #endif

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, X, Y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET))) {
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

	if ((retCode = checkVectorSizes(kargs->dtype, N, Y, offy, incy, Y_VEC_ERRSET))) {
        printf("Invalid Size for Y\n");
        return retCode;
    }

    if ((commandQueue == NULL) || (numCommandQueues == 0))
    {
        return clblasInvalidValue;
    }

    if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
    {
        return clblasInvalidEventWaitList;
    }

    kargs->order = order;
	if(order == clblasRowMajor)		// Handling row-major. Invert X, Y and uplo
	{
		kargs->uplo = (uplo == clblasUpper) ? clblasLower : clblasUpper;
		kargs->B = Y;
		kargs->ldb.vector = incy;
		kargs->offBX = offy;
		kargs->C = X;
		kargs->ldc.vector = incx;
		kargs->offCY = offx;
	}
	else
	{
		kargs->uplo = uplo;
		kargs->B = X;
		kargs->ldb.vector = incx;
		kargs->offBX = offx;
		kargs->C = Y;
		kargs->ldc.vector = incy;
		kargs->offCY = offy;
	}
    kargs->N = N;
    kargs->A = A;
    kargs->lda.matrix = lda;
    kargs->offa = offa;
	kargs->offA = offa;

    #ifdef DEBUG_HER2
    printf("Calling makeSolutionSeq : HER2\n");
    #endif

    /*
     * Always use CommandQueue (0)
     * PENDING:
     * 1. No Multi-GPU / Multi-command queue support
     * 2. This can be optimized to use the commandQ with the higher
     *    memmory bandwidth that supports the data-type and the LDA
     */
    numCommandQueues = 1;

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_HER2, kargs, numCommandQueues, commandQueue,
                          numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);
    return (clblasStatus)err;
}

clblasStatus
clblasCher2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
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
    cl_command_queue* commandQueue,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
	{
		CLBlasKargs kargs;

        memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_FLOAT;
        kargs.alpha.argFloatComplex = alpha;
        kargs.pigFuncID = CLBLAS_HER2;

		#ifdef DEBUG_HER2
		printf("Cher2 called\n");
		#endif

		return doHer2(&kargs, order, uplo, N, X, offx, incx, Y, offy, incy, A, offa, lda,
						numCommandQueues, commandQueue, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZher2(
    clblasOrder order,
    clblasUplo uplo,
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
    cl_command_queue* commandQueue,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
    {
		CLBlasKargs kargs;

        memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_DOUBLE;
        kargs.alpha.argDoubleComplex = alpha;
        kargs.pigFuncID = CLBLAS_HER2;

        #ifdef DEBUG_HER2
        printf("Zher2 called\n");
        #endif

        return doHer2(&kargs, order, uplo, N, X, offx, incx, Y, offy, incy, A, offa, lda,
                        numCommandQueues, commandQueue, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clblasChpr2(
	clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueue,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
	{
		CLBlasKargs kargs;

        memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_FLOAT;
        kargs.alpha.argFloatComplex = alpha;
        kargs.pigFuncID = CLBLAS_HPR2;

		return doHer2(&kargs, order, uplo, N, X, offx, incx, Y, offy, incy, AP, offa, 0,
						numCommandQueues, commandQueue, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clblasZhpr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueue,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events)
    {
		CLBlasKargs kargs;

        memset(&kargs, 0, sizeof(kargs));
        kargs.dtype = TYPE_COMPLEX_DOUBLE;
        kargs.alpha.argDoubleComplex = alpha;
        kargs.pigFuncID = CLBLAS_HPR2;

        return doHer2(&kargs, order, uplo, N, X, offx, incx, Y, offy, incy, AP, offa, 0,
                        numCommandQueues, commandQueue, numEventsInWaitList, eventWaitList, events);
    }

