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
doTrmm(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    cl_int err;
    ListHead seq;
    size_t msize;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    /* Validate arguments */

    if ((retCode = checkMemObjects(A, B, NULL, false, A_MAT_ERRSET, B_MAT_ERRSET, END_ERRSET))) {
        return retCode;
    }
    msize = (side == clblasLeft) ? M : N;
    if ((retCode = checkMatrixSizes(kargs->dtype, order, transA, msize, msize,
                                    A, offA, lda, A_MAT_ERRSET ))) {
        return retCode;
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, M, N,
                                    B, offB, ldb, B_MAT_ERRSET ))) {
        return retCode;
    }

    kargs->order = order;
    kargs->side = side;
    kargs->uplo = uplo;
    kargs->transA = transA;
    kargs->diag = diag;
    kargs->M = M;
    kargs->N = N;
    kargs->A = A;
    kargs->offA = offA;
    kargs->lda.matrix = lda;
    kargs->B = B;
    kargs->offBX = offB;
    kargs->ldb.matrix = ldb;
    // Store original problem size in K, this is used to know it while
    // calculating result by parts using M or N as part size
    if (side == clblasLeft) {
        kargs->K = M;
    }
    else {
        kargs->K = N;
    }

    kargs->offsetM = 0;
    kargs->offsetN = 0;
    kargs->scimage[0] = 0;

#ifndef TRXM_MULTIPLE_QUEUES
    if (numCommandQueues != 0) {
        numCommandQueues = 1;
    }
#endif

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_TRMM, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);

    return (clblasStatus)err;
}

clblasStatus
clblasStrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
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

    return doTrmm(&kargs, order, side, uplo, transA, diag, M, N, A, offA, lda,
                  B, offB, ldb, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDtrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
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

    return doTrmm(&kargs, order, side, uplo, transA, diag, M, N, A, offA, lda,
                  B, offB, ldb, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCtrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
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

    return doTrmm(&kargs, order, side, uplo, transA, diag, M, N, A, offA, lda,
                  B, offB, ldb, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZtrmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
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

    return doTrmm(&kargs, order, side, uplo, transA, diag, M, N, A, offA, lda,
                  B, offB, ldb, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}
