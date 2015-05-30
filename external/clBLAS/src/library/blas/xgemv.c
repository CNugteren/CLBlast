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
doGemv(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
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
    size_t sizev;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects( A, x, y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET ))) {
        return retCode;
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans,
                                    M, N, A, offA, lda, A_MAT_ERRSET ))) {
        return retCode;
    }
    sizev = (transA == clblasNoTrans) ? N : M;
    if ((retCode = checkVectorSizes(kargs->dtype, sizev, x, offx, incx, X_VEC_ERRSET ))) {
        return retCode;
    }
    sizev = (transA == clblasNoTrans) ? M : N;
    if ((retCode = checkVectorSizes(kargs->dtype, sizev, y, offy, incy, Y_VEC_ERRSET))) {
        return retCode;
    }

    kargs->order = order;
    kargs->transA = transA;
    kargs->M = M;
    kargs->N = N;
    /*
     * store original height of the matrix A
     * FIXME: store it to a dedicated field
     */
    kargs->K = (transA == clblasNoTrans) ? M : N;
    kargs->A = A;
    kargs->offA = offA;
    kargs->lda.matrix = lda;
    kargs->B = x;
    kargs->offBX = offx;
    kargs->ldb.vector = incx;
    kargs->C = y;
    kargs->offCY = offy;
    kargs->ldc.vector = incy;

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_GEMV, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);

    return (clblasStatus)err;
}

clblasStatus
clblasSgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
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

    return doGemv(&kargs, order, transA, M, N, A, offA, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
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

    return doGemv(&kargs, order, transA, M, N, A, offA, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
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

    return doGemv(&kargs, order, transA, M, N, A, offA, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZgemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
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

    return doGemv(&kargs, order, transA, M, N, A, offA, lda, x, offx, incx,
                  y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}
