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


#include <string.h>
#include <clBLAS.h>
#include <devinfo.h>

#include "clblas-internal.h"
#include "solution_seq.h"

static clblasStatus
doHemv(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
	size_t N,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err;
    ListHead seq1, seq2;
	cl_event first_event;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, x, y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, N, N,
                                    A, offA, lda, A_MAT_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, x, offx, incx, X_VEC_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, y, offy, incy, Y_VEC_ERRSET))) {
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

	numCommandQueues = 1;
    kargs->order = order;
    kargs->uplo = uplo;
    kargs->N = N;
    kargs->A = A;
    kargs->offA = offA;
	kargs->offa = offA;
    kargs->lda.matrix = lda;
    kargs->B = x;
    kargs->offBX = offx;
    kargs->ldb.vector = incx;
    kargs->C = y;
    kargs->offCY = offy;
    kargs->ldc.vector = incy;
	kargs->transA = clblasNoTrans;
	kargs->diag = clblasNonUnit;

	listInitHead(&seq1);
    err = makeSolutionSeq(CLBLAS_HEMV, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, &first_event, &seq1);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq1);
		if (err == CL_SUCCESS)
		{
			listInitHead(&seq2);
			kargs->transA = clblasConjTrans;
		    kargs->diag   = clblasUnit;
			err = makeSolutionSeq(CLBLAS_HEMV, kargs, numCommandQueues, commandQueues,
			       1, &first_event, events, &seq2);
			if (err == CL_SUCCESS)
			{
				err = executeSolutionSeq(&seq2);
			}
			freeSolutionSeq(&seq2);
		}
    }

    freeSolutionSeq(&seq1);
    return (clblasStatus)err;

	//printf("doHemv called\n");
	//return 0;
}

clblasStatus
clblasChemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
	size_t offa,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = beta;

    return doHemv(&kargs, order, uplo, N, A, offa, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZhemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
	size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = beta;

    return doHemv(&kargs, order, uplo, N, A, offa, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

