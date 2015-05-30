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

clblasStatus
doSyr2k(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_mem C,
    size_t offC,
    size_t ldc,
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

    if (numCommandQueues == 0 || commandQueues == NULL) {
        return clblasInvalidValue;
    }

    // Validate arguments
    if ((retCode = checkMemObjects(A, B, C, true, A_MAT_ERRSET, B_MAT_ERRSET, C_MAT_ERRSET))) {
        return retCode;
    }

    if (isComplexType(kargs->dtype) && transAB == clblasConjTrans) {
        return clblasInvalidValue;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, transAB, N, K, A, offA, lda, A_MAT_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, transAB, N, K, B, offB, ldb, B_MAT_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkMatrixSizes(kargs->dtype, order, false, N, N, C, offC, ldc, C_MAT_ERRSET))) {
        return retCode;
    }

    kargs->order = order;
    kargs->transA = transAB;
    kargs->transB = transAB;
    kargs->uplo = uplo;
    kargs->M = N;
    kargs->N = N;
    kargs->K = K;
    kargs->A = A;
    kargs->offA = offA;
    kargs->lda.matrix = lda;
    kargs->B = B;
    kargs->offBX = offB;
    kargs->ldb.matrix = ldb;
    kargs->C = C;
    kargs->offCY = offC;
    kargs->ldc.matrix = ldc;

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_SYR2K, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);

    return (clblasStatus)err;
}

clblasStatus
clblasSsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.alpha.argFloat = alpha;
    kargs.beta.argFloat = beta;
    kargs.dtype = TYPE_FLOAT;

    return doSyr2k(&kargs, order, uplo, transAB, N, K, A, offA, lda, B, offB,
                   ldb, C, offC, ldc, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.alpha.argDouble = alpha;
    kargs.beta.argDouble = beta;
    kargs.dtype = TYPE_DOUBLE;

    return doSyr2k(&kargs, order, uplo, transAB, N, K, A, offA, lda, B, offB,
                   ldb, C, offC, ldc, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasCsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = beta;
    kargs.dtype = TYPE_COMPLEX_FLOAT;

    return doSyr2k(&kargs, order, uplo, transAB, N, K, A, offA, lda, B, offB,
                   ldb, C, offC, ldc, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZsyr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = beta;
    kargs.dtype = TYPE_COMPLEX_DOUBLE;

    return doSyr2k(&kargs, order, uplo, transAB, N, K, A, offA, lda, B, offB,
                   ldb, C, offC, ldc, numCommandQueues, commandQueues,
                   numEventsInWaitList, eventWaitList, events);
}
