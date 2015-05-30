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


//#define USE_SYMV

#include <string.h>
#include <clBLAS.h>
#include <devinfo.h>

#include "clblas-internal.h"
#include "solution_seq.h"

static clblasStatus
doSymv(
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
    ListHead seq;
    clblasStatus retCode = clblasSuccess;
    #ifdef USE_SYMV
        ListHead seq2;
        ListNode *listNodePtr;
	    cl_event first_event;
    #endif

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, x, y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, N, N,
                                    A, offA, lda, A_MAT_ERRSET ))) {
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, x, offx, incx, X_VEC_ERRSET ))) {
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, y, offy, incy, Y_VEC_ERRSET ))) {
        return retCode;
    }

    kargs->order = order;
    kargs->uplo = uplo;
    kargs->N = N;
    kargs->K = N; //store original N
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

    #ifndef USE_SYMV

        listInitHead(&seq);
        err = makeSolutionSeq(CLBLAS_SYMV, kargs, numCommandQueues, commandQueues,
            numEventsInWaitList, eventWaitList, events, &seq);
        if (err == CL_SUCCESS) {
            err = executeSolutionSeq(&seq);
        }

    #else   // version of SYMV using kprintf

        numCommandQueues = 1;
        listInitHead(&seq);

	    kargs->transA = clblasNoTrans;
	    kargs->diag = clblasNonUnit;
		err = makeSolutionSeq(CLBLAS_HEMV, kargs, numCommandQueues, commandQueues,
        					  numEventsInWaitList, eventWaitList, &first_event, &seq);
		if (err == CL_SUCCESS)
        {
            listInitHead(&seq2);

			kargs->transA = clblasTrans;
		    kargs->diag   = clblasUnit;
            err = makeSolutionSeq(CLBLAS_HEMV, kargs, numCommandQueues, commandQueues,
                       1, &first_event, events, &seq2);

            if (err == CL_SUCCESS)
            {
                // Adding node from seq2 to main seq
                listNodePtr = listNodeFirst(&seq2);
                listAddToTail(&seq, listNodePtr);

                err = executeSolutionSeq(&seq);     // Executes both kernels in the seq one after other
            }
		}

    #endif

    freeSolutionSeq(&seq);
    return (clblasStatus)err;
}

clblasStatus
clblasSsymv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
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
    kargs.dtype = TYPE_FLOAT;
    kargs.alpha.argFloat = alpha;
    kargs.beta.argFloat = beta;

    return doSymv(&kargs, order, uplo, N, A, offA, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDsymv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
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
    kargs.dtype = TYPE_DOUBLE;
    kargs.alpha.argDouble = alpha;
    kargs.beta.argDouble = beta;

    return doSymv(&kargs, order, uplo, N, A, offA, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}
