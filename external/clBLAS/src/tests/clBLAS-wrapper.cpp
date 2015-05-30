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


#include <clBLAS.h>
#include "clBLAS-wrapper.h"

clblasStatus
clMath::clblas::gemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasSgemv(order, transA, M, N, alpha, A, offA, lda, X,
                            offx, incx, beta, Y, offy, incy,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret =  clblasDgemv(order, transA, M, N, alpha, A, offA, lda, X,
                            offx, incx, beta, Y, offy, incy,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasCgemv(order, transA, M, N, alpha, A, offA, lda, X,
                            offx, incx, beta, Y, offy, incy,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasZgemv(order, transA, M, N, alpha, A, offA, lda, X, offx,
                            incx, beta, Y, offy, incy, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

// SYMV wrappers
clblasStatus
clMath::clblas::symv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret =  clblasSsymv(order, uplo, N, alpha, A, offA, lda, X, offx,
                            incx, beta, Y, offy, incy, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::symv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasDsymv(order, uplo, N, alpha, A, offA, lda, X, offx,
                            incx, beta, Y, offy, incy, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasSgemm(order, transA, transB, M, N, K, alpha, A, offA,
                            lda, B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasDgemm(order, transA, transB, M, N, K, alpha, A, offA,
                            lda, B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
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
    clblasStatus ret;

    ret = clblasCgemm(order, transA, transB, M, N, K, alpha, A, offA,
                            lda, B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::gemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
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
    clblasStatus ret;

    ret = clblasZgemm(order, transA, transB, M, N, K, alpha, A, offA,
                            lda, B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

#undef GEMMV2_VISIBLE // GEMM2 is not exported.

clblasStatus
clMath::clblas::gemm2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret = clblasNotImplemented;

#ifdef GEMMV2_VISIBLE //If GEMM2 is visible

    if (!(offA || offB || offC)) {
        ret = clblasSgemmV2(order, transA, transB, M, N, K, alpha, A, lda,
                             B, ldb, beta, C, ldc, numCommandQueues,
                             commandQueues, numEventsInWaitList, eventWaitList,
                             events);
    }
    else {
        ret = clblasSgemmExV2(order, transA, transB, M, N, K, alpha, A, offA,
                               lda, B, offB, ldb, beta, C, offC, ldc,
                               numCommandQueues, commandQueues,
                               numEventsInWaitList, eventWaitList, events);
    }
#else //To avoid warnings
    order = order;
    transA = transA;
    transB = transB;
    M = M;
    N = N;
    K = K;
    alpha = alpha;
    lda = lda;
    ldb = ldb;
    beta = beta;
    C = A;
    C = B;
    C = C;
    ldc = ldc;
    numCommandQueues = numCommandQueues;
    commandQueues = commandQueues;
    numEventsInWaitList = numEventsInWaitList;
    eventWaitList = eventWaitList;
    events = events;
    offA = offA;
    offB = offB;
    offC = offC;
#endif

    return ret;
}

clblasStatus
clMath::clblas::gemm2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret = clblasNotImplemented;

#ifdef GEMMV2_VISIBLE
    if (!(offA || offB || offC)) {
        ret = clblasDgemmV2(order, transA, transB, M, N, K, alpha, A, lda,
                             B, ldb, beta, C, ldc, numCommandQueues,
                             commandQueues, numEventsInWaitList, eventWaitList,
                             events);
    }
    else {
        ret = clblasDgemmExV2(order, transA, transB, M, N, K, alpha, A, offA,
                               lda, B, offB, ldb, beta, C, offC, ldc,
                               numCommandQueues, commandQueues,
                               numEventsInWaitList, eventWaitList, events);
    }
#else //To avoid warnings
    order = order;
    transA = transA;
    transB = transB;
    M = M;
    N = N;
    K = K;
    alpha = alpha;
    lda = lda;
    ldb = ldb;
    beta = beta;
    C = A;
    C = B;
    C = C;
    ldc = ldc;
    numCommandQueues = numCommandQueues;
    commandQueues = commandQueues;
    numEventsInWaitList = numEventsInWaitList;
    eventWaitList = eventWaitList;
    events = events;
    offA = offA;
    offB = offB;
    offC = offC;
#endif

    return ret;
}

clblasStatus
clMath::clblas::gemm2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
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
    clblasStatus ret = clblasNotImplemented;

#ifdef GEMMV2_VISIBLE
    if (!(offA || offB || offC)) {
        ret = clblasCgemmV2(order, transA, transB, M, N, K, alpha, A, lda,
                             B, ldb, beta, C, ldc, numCommandQueues,
                             commandQueues, numEventsInWaitList, eventWaitList,
                             events);
    }
    else {
        ret = clblasCgemmExV2(order, transA, transB, M, N, K, alpha, A, offA,
                               lda, B, offB, ldb, beta, C, offC, ldc,
                               numCommandQueues, commandQueues,
                               numEventsInWaitList, eventWaitList, events);
    }
#else //To avoid warnings
    order = order;
    transA = transA;
    transB = transB;
    M = M;
    N = N;
    K = K;
    alpha = alpha;
    lda = lda;
    ldb = ldb;
    beta = beta;
    C = A;
    C = B;
    C = C;
    ldc = ldc;
    numCommandQueues = numCommandQueues;
    commandQueues = commandQueues;
    numEventsInWaitList = numEventsInWaitList;
    eventWaitList = eventWaitList;
    events = events;
    offA = offA;
    offB = offB;
    offC = offC;
#endif
    return ret;
}

clblasStatus
clMath::clblas::gemm2(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
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
    clblasStatus ret = clblasNotImplemented;

#ifdef GEMMV2_VISIBLE
    if (!(offA || offB || offC)) {
        ret = clblasZgemmV2(order, transA, transB, M, N, K, alpha, A, lda,
                             B, ldb, beta, C, ldc, numCommandQueues,
                             commandQueues, numEventsInWaitList, eventWaitList,
                             events);
    }
    else {
        ret = clblasZgemmExV2(order, transA, transB, M, N, K, alpha, A, offA,
                               lda, B, offB, ldb, beta, C, offC, ldc,
                               numCommandQueues, commandQueues,
                               numEventsInWaitList, eventWaitList, events);
    }

#else //To avoid warnings
    order = order;
    transA = transA;
    transB = transB;
    M = M;
    N = N;
    K = K;
    alpha = alpha;
    lda = lda;
    ldb = ldb;
    beta = beta;
    C = A;
    C = B;
    C = C;
    ldc = ldc;
    numCommandQueues = numCommandQueues;
    commandQueues = commandQueues;
    numEventsInWaitList = numEventsInWaitList;
    eventWaitList = eventWaitList;
    events = events;
    offA = offA;
    offB = offB;
    offC = offC;
#endif

    return ret;
}

clblasStatus
clMath::clblas::trmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    float alpha,
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
    clblasStatus ret;

    ret = clblasStrmm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    double alpha,
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
    clblasStatus ret;

    ret = clblasDtrmm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trmm(
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
    clblasStatus ret;

    ret = clblasCtrmm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trmm(
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
    clblasStatus ret;

    ret = clblasZtrmm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    float alpha,
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
    clblasStatus ret;

    ret = clblasStrsm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    double alpha,
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
    clblasStatus ret;

    ret = clblasDtrsm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trsm(
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
    clblasStatus ret;

    ret = clblasCtrsm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trsm(
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
    clblasStatus ret;

    ret = clblasZtrsm(order, side, uplo, transA, diag, M, N, alpha, A,
                            offA, lda, B, offB, ldb, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasSsyr2k(order, uplo, transAB, N, K, alpha, A, offA, lda,
                            B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transAB,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasDsyr2k(order, uplo, transAB, N, K, alpha, A, offA, lda,
                            B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syr2k(
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
    clblasStatus ret;

    ret = clblasCsyr2k(order, uplo, transAB, N, K, alpha, A, offA, lda,
                            B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syr2k(
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
    clblasStatus ret;

    ret = clblasZsyr2k(order, uplo, transAB, N, K, alpha, A, offA, lda,
                            B, offB, ldb, beta, C, offC, ldc,
                            numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasSsyrk(order, uplo, transA, N, K, alpha, A, offA, lda,
                            beta, C, offC, ldc, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasDsyrk(order, uplo, transA, N, K, alpha, A, offA, lda,
                            beta, C, offC, ldc, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
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
    clblasStatus ret;

    ret = clblasCsyrk(order, uplo, transA, N, K, alpha, A, offA, lda,
                            beta, C, offC, ldc, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::syrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
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
    clblasStatus ret;

    ret = clblasZsyrk(order, uplo, transA, N, K, alpha, A, offA, lda,
                            beta, C, offC, ldc, numCommandQueues,
                            commandQueues, numEventsInWaitList,
                            eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::trmv(
	DataType type,
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
		switch(type)
		{
			case TYPE_FLOAT:
				return clblasStrmv(
					order, uplo, trans, diag, N, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues,commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_DOUBLE:
				return clblasDtrmv(
					order, uplo, trans, diag, N, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_COMPLEX_FLOAT:
				return clblasCtrmv(
					order, uplo, trans, diag, N, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_COMPLEX_DOUBLE:
				return clblasZtrmv(
					order, uplo, trans, diag, N, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			default:
				return 	clblasInvalidValue;
		}
	}

clblasStatus
clMath::clblas::trsv(
    DataType type,
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
	switch(type)
        {
            case TYPE_FLOAT:
                return clblasStrsv(
                    order, uplo, trans, diag, N, A, offa, lda, X,
                    offx, incx, numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDtrsv(
                    order, uplo, trans, diag, N, A, offa, lda, X,
                    offx, incx, numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
                return clblasCtrsv(
                    order, uplo, trans, diag, N, A, offa, lda, X,
                    offx, incx,numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
                return clblasZtrsv(
                    order, uplo, trans, diag, N, A, offa, lda, X,
                    offx, incx, numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            default:
                return  clblasInvalidValue;
        }
    }

clblasStatus
clMath::clblas::tpsv(
    DataType type,
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
    switch(type)
        {
            case TYPE_FLOAT:
                return clblasStpsv(
                    order, uplo, trans, diag, N, A, offa, X,
                    offx, incx, numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDtpsv(
                    order, uplo, trans, diag, N, A, offa, X,
                    offx, incx, numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
                return clblasCtpsv(
                    order, uplo, trans, diag, N, A, offa, X,
                    offx, incx,numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
                return clblasZtpsv(
                    order, uplo, trans, diag, N, A, offa, X,
                    offx, incx, numCommandQueues,
                    commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            default:
                return  clblasInvalidValue;
        }
    }

clblasStatus
clMath::clblas::symm(
	clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
	{
		return clblasSsymm( order, side, uplo, M, N, alpha,
    							A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
    							numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}


clblasStatus
clMath::clblas::symm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasDsymm( order, side, uplo, M, N, alpha,
                                A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::symm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasCsymm( order, side, uplo, M, N, alpha,
                                A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::symm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZsymm( order, side, uplo, M, N, alpha,
                                A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::ger(
    clblasOrder order,
    size_t M,
    size_t N,
    float alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
        {
                return clblasSger( order, M, N, alpha,
                                        X, offx, incx, Y, offy, incy, A, offa, lda,
                                        numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
        }


clblasStatus
clMath::clblas::ger(
    clblasOrder order,
    size_t M,
    size_t N,
    double alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasDger( order, M, N, alpha,
                                X, offx, incx, Y, offy, incy, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::ger(
    clblasOrder order,
    size_t M,
    size_t N,
    FloatComplex alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasCgeru( order, M, N, alpha,
                                X, offx, incx, Y, offy, incy, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::ger(
    clblasOrder order,
    size_t M,
    size_t N,
    DoubleComplex alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZgeru( order, M, N, alpha,
                                X, offx, incx, Y, offy, incy, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::gerc(
    clblasOrder order,
    size_t M,
    size_t N,
    FloatComplex alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasCgerc( order, M, N, alpha,
                                X, offx, incx, Y, offy, incy, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::gerc(
    clblasOrder order,
    size_t M,
    size_t N,
    DoubleComplex alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZgerc( order, M, N, alpha,
                                X, offx, incx, Y, offy, incy, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::syr(
	clblasOrder order,
	clblasUplo uplo,
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
	cl_uint numCommandQueues,
	cl_command_queue *commandQueues,
	cl_uint numEventsInWaitList,
	const cl_event *eventWaitList,
	cl_event *events)
	{
		return clblasSsyr( order, uplo, N, alpha, X, offx, incx, A, offa, lda,
								numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clMath::clblas::syr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasDsyr( order, uplo, N, alpha, X, offx, incx, A, offa, lda,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

//SPR

clblasStatus
clMath::clblas::her(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasCher( order, uplo, N, alpha,
                                X, offx, incx, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::her(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZher( order, uplo, N, alpha,
                                X, offx, incx, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::syr2(
	clblasOrder order,
	clblasUplo uplo,
    size_t N,
    float alpha,
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
	cl_command_queue *commandQueues,
	cl_uint numEventsInWaitList,
	const cl_event *eventWaitList,
	cl_event *events)
	{
		return clblasSsyr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
								numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clMath::clblas::syr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasDsyr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::her2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    FloatComplex alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasCher2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::her2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    DoubleComplex alpha,
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
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZher2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, A, offa, lda,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::hemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    FloatComplex alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    FloatComplex beta,
        cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasChemv( order, uplo, N, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::hemv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZhemv( order, uplo, N, alpha, A, offa, lda, X, offx, incx, beta, Y, offy, incy,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


//HEMM
clblasStatus
clMath::clblas::hemm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasChemm( order, side, uplo, M, N, alpha,
                                A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::hemm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
	 size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZhemm( order, side, uplo, M, N, alpha,
                                A, offa, lda, B, offb, ldb, beta, C, offc, ldc,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::herk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	clblasStatus ret;

	ret = clblasCherk(order, uplo, transA, N, K, alpha, A, offA, lda,
						beta, C, offC, ldc, numCommandQueues,
                        commandQueues, numEventsInWaitList,
                        eventWaitList, events);

	return ret;
}

clblasStatus
clMath::clblas::herk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasZherk(order, uplo, transA, N, K, alpha, A, offA, lda,
                        beta, C, offC, ldc, numCommandQueues,
                        commandQueues, numEventsInWaitList,
                        eventWaitList, events);

    return ret;
}


clblasStatus
clMath::clblas::tpmv(
	DataType type,
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
		switch(type)
		{
			case TYPE_FLOAT:
				return clblasStpmv(
					order, uplo, trans, diag, N, AP, offa, X,
					offx, incx, scratchBuff,
					numCommandQueues,commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_DOUBLE:
				return clblasDtpmv(
					order, uplo, trans, diag, N, AP, offa, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_COMPLEX_FLOAT:
				return clblasCtpmv(
					order, uplo, trans, diag, N, AP, offa, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_COMPLEX_DOUBLE:
				return clblasZtpmv(
					order, uplo, trans, diag, N, AP, offa, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			default:
				return 	clblasInvalidValue;
		}
	}


clblasStatus
clMath::clblas::spmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret =  clblasSspmv(order, uplo, N, alpha, AP, offa, X, offx, incx,
                          beta, Y, offy, incy, numCommandQueues,
                          commandQueues, numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::spmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasDspmv(order, uplo, N, alpha, AP, offa, X, offx, incx, beta,
                         Y, offy, incy, numCommandQueues, commandQueues,
                         numEventsInWaitList, eventWaitList, events);
    return ret;
}

clblasStatus
clMath::clblas::hpmv(
      clblasOrder order,
      clblasUplo uplo,
      size_t N,
      FloatComplex alpha,
      const cl_mem AP,
      size_t offa,
      const cl_mem X,
      size_t offx,
      int incx,
      FloatComplex beta,
      cl_mem Y,
      size_t offy,
      int incy,
      cl_uint numCommandQueues,
      cl_command_queue *commandQueues,
      cl_uint numEventsInWaitList,
      const cl_event *eventWaitList,
      cl_event *events)
{

        return clblasChpmv(order, uplo, N, alpha, AP, offa,
                                    X, offx, incx, beta, Y, offy, incy,
                                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}

clblasStatus
clMath::clblas::hpmv(
      clblasOrder order,
      clblasUplo uplo,
      size_t N,
      DoubleComplex alpha,
      const cl_mem AP,
      size_t offa,
      const cl_mem X,
      size_t offx,
      int incx,
      DoubleComplex beta,
      cl_mem Y,
      size_t offy,
      int incy,
      cl_uint numCommandQueues,
      cl_command_queue *commandQueues,
      cl_uint numEventsInWaitList,
      const cl_event *eventWaitList,
      cl_event *events)
{

        return clblasZhpmv(order, uplo, N, alpha, AP, offa,
                                    X, offx, incx, beta, Y, offy, incy,
                                    numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events );
}



clblasStatus
clMath::clblas::spr(
	clblasOrder order,
	clblasUplo uplo,
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
	cl_uint numCommandQueues,
	cl_command_queue *commandQueues,
	cl_uint numEventsInWaitList,
	const cl_event *eventWaitList,
	cl_event *events)
	{
		return clblasSspr( order, uplo, N, alpha, X, offx, incx, AP, offa,
								numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clMath::clblas::spr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasDspr( order, uplo, N, alpha, X, offx, incx, AP, offa,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::hpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasChpr( order, uplo, N, alpha,
                                X, offx, incx, AP, offa,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::hpr(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZhpr( order, uplo, N, alpha,
                                X, offx, incx, AP, offa,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }


clblasStatus
clMath::clblas::spr2(
	clblasOrder order,
	clblasUplo uplo,
    size_t N,
    float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
	cl_uint numCommandQueues,
	cl_command_queue *commandQueues,
	cl_uint numEventsInWaitList,
	const cl_event *eventWaitList,
	cl_event *events)
	{
		return clblasSspr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
								numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
	}

clblasStatus
clMath::clblas::spr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasDspr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::hpr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    FloatComplex alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasChpr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
                                numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::hpr2(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    DoubleComplex alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
    {
        return clblasZhpr2( order, uplo, N, alpha, X, offx, incx, Y, offy, incy, AP, offa,
                              numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
    }

clblasStatus
clMath::clblas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_float alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasSgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }

clblasStatus
clMath::clblas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_double alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasDgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }

clblasStatus
clMath::clblas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_float2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasCgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }

clblasStatus
clMath::clblas::gbmv(
        clblasOrder order,
        clblasTranspose trans,
        size_t M,
        size_t N,
        size_t KL,
        size_t KU,
        cl_double2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasZgbmv( order, trans, M, N, KL, KU, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }

clblasStatus
clMath::clblas::tbmv(
        DataType type,
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
        switch(type)
		{
			case TYPE_FLOAT:
				return clblasStbmv(
					order, uplo, trans, diag, N, K, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues,commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_DOUBLE:
				return clblasDtbmv(
					order, uplo, trans, diag, N, K, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_COMPLEX_FLOAT:
				return clblasCtbmv(
					order, uplo, trans, diag, N, K, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			case TYPE_COMPLEX_DOUBLE:
				return clblasZtbmv(
					order, uplo, trans, diag, N, K, A, offa, lda, X,
					offx, incx, scratchBuff,
					numCommandQueues, commandQueues, numEventsInWaitList,
					eventWaitList, events);

			default:
				return 	clblasInvalidValue;
		}
    }

//SBMV

clblasStatus
clMath::clblas::sbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_float alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasSsbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events );
        }

clblasStatus
clMath::clblas::sbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t M,
        size_t K,
        cl_double alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasDsbmv( order, uplo, M, K, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }


//HBMV

clblasStatus
clMath::clblas::hbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_float2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_float2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasChbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }

clblasStatus
clMath::clblas::hbmv(
        clblasOrder order,
        clblasUplo uplo,
        size_t N,
        size_t K,
        cl_double2 alpha,
        const cl_mem A,
        size_t offa,
        size_t lda,
        const cl_mem X,
        size_t offx,
        int incx,
        cl_double2 beta,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
        {
            return clblasZhbmv( order, uplo, N, K, alpha, A, offa, lda, X, offx, incx,
                                    beta, Y, offy, incy, numCommandQueues, commandQueues,
                                        numEventsInWaitList, eventWaitList, events);
        }

//TBSV

clblasStatus
clMath::clblas::tbsv(
        DataType type,
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
        //cl_mem scratchBuff,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
        switch(type)
        {
            case TYPE_FLOAT:
                return clblasStbsv(
                    order, uplo, trans, diag, N, K, A, offa, lda, X,
                    offx, incx,
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDtbsv(
                    order, uplo, trans, diag, N, K, A, offa, lda, X,
                    offx, incx,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
                return clblasCtbsv(
                    order, uplo, trans, diag, N, K, A, offa, lda, X,
                    offx, incx,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
                return clblasZtbsv(
                    order, uplo, trans, diag, N, K, A, offa, lda, X,
                    offx, incx,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            default:
                return  clblasInvalidValue;
        }
    }


clblasStatus
clMath::clblas::her2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
	clblasStatus ret;

	ret = clblasCher2k(order, uplo, transA, N, K, alpha, A, offa, lda, B, offb, ldb,
						beta, C, offc, ldc, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);

	return ret;
}

clblasStatus
clMath::clblas::her2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus ret;

    ret = clblasZher2k(order, uplo, transA, N, K, alpha, A, offa, lda, B, offb, ldb,
						beta, C, offc, ldc, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);

    return ret;
}

clblasStatus
clMath::clblas::swap(
	DataType type,
    size_t N,
    cl_mem X,
	size_t offx,
    int incx,
	cl_mem Y,
	size_t offy,
    int incy,
	cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
		switch(type)
		{
			case TYPE_FLOAT:
				return clblasSswap(
					N, X, offx, incx, Y, offy, incy,
					numCommandQueues, commandQueues, numEventsInWaitList,
				    eventWaitList, events);

			case TYPE_DOUBLE:
				return clblasDswap(
					N, X, offx, incx, Y, offy, incy,
					numCommandQueues, commandQueues, numEventsInWaitList,
				    eventWaitList, events);

			case TYPE_COMPLEX_FLOAT:
				return clblasCswap(
					N, X, offx, incx, Y, offy, incy,
					numCommandQueues, commandQueues, numEventsInWaitList,
				    eventWaitList, events);

			case TYPE_COMPLEX_DOUBLE:
				return clblasZswap(
					N, X, offx, incx, Y, offy, incy,
					numCommandQueues, commandQueues, numEventsInWaitList,
				    eventWaitList, events);

			default:
				return 	clblasInvalidValue;
		}
}

 clblasStatus
clMath::clblas::copy(
	DataType type,
    size_t N,
    cl_mem X,
	size_t offx,
    int incx,
	cl_mem Y,
	size_t offy,
    int incy,
	cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
   {
        switch(type)
        {

            case TYPE_FLOAT:
                return clblasScopy(
                    N, X, offx, incx, Y,
                    offy, incy,
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDcopy(
                    N, X, offx, incx, Y,
                    offy, incy,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
                return clblasCcopy(
                    N, X, offx, incx, Y,
                    offy, incy,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                     eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
                return clblasZcopy(
                    N, X, offx, incx, Y,
                    offy, incy,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            default:
                return  clblasInvalidValue;
        }
    }


// scal, csscal & zdscal wrappers
clblasStatus
clMath::clblas::scal(
        bool is_css_zds,
        size_t N,
        cl_float alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{
    is_css_zds = is_css_zds;     // Remove warning
    return clblasSscal(N, alpha, X, offx, incx, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clMath::clblas::scal(
        bool is_css_zds,
        size_t N,
        cl_double alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{
    is_css_zds = is_css_zds;     // Remove warning
    return clblasDscal(N, alpha, X, offx, incx, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clMath::clblas::scal(
        bool is_css_zds,
        size_t N,
        FloatComplex alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{

    if(is_css_zds) {
        return clblasCsscal(N, CREAL(alpha), X, offx, incx, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
    } else {
        return clblasCscal(N, alpha, X, offx, incx, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
    }
}

clblasStatus
clMath::clblas::scal(
        bool is_css_zds,
        size_t N,
        DoubleComplex alpha,
        cl_mem X,
        size_t offx,
        int incx,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{
    if(is_css_zds) {
        return clblasZdscal(N, CREAL(alpha), X, offx, incx, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
    } else {
        return clblasZscal(N, alpha, X, offx, incx, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
    }
}

// DOT
clblasStatus
clMath::clblas::dot(
    DataType type,
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)

    {
            switch(type){

            case TYPE_FLOAT:
            return clblasSdot( N, dotProduct, offDP, X, offx, incx, Y,
                            offy, incy, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_DOUBLE:
            return clblasDdot( N, dotProduct, offDP, X, offx, incx, Y,
                                  offy, incy, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
            return clblasCdotu( N, dotProduct, offDP, X, offx, incx, Y,
                            offy, incy, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
            return clblasZdotu( N, dotProduct, offDP, X, offx, incx, Y,
                                  offy, incy, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            default:
                   return clblasInvalidValue;
            }

    }


//ASUM

clblasStatus
clMath::clblas::asum(
    DataType type,
    size_t N,
    cl_mem asum,
    size_t offAsum,
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
            switch(type){

            case TYPE_FLOAT:
            return clblasSasum( N, asum, offAsum, X, offx, incx, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_DOUBLE:
            return clblasDasum( N, asum, offAsum, X, offx, incx, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
            return clblasScasum( N, asum, offAsum, X, offx, incx, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
            return clblasDzasum( N, asum, offAsum, X, offx, incx, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            default:
                   return clblasInvalidValue;
            }

    }

//DOTC
clblasStatus
clMath::clblas::dotc(
    DataType type,
    size_t N,
    cl_mem dotProduct,
    size_t offDP,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)

    {
            switch(type){

            case TYPE_COMPLEX_FLOAT:
            return clblasCdotc( N, dotProduct, offDP, X, offx, incx, Y,
                            offy, incy, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
            return clblasZdotc( N, dotProduct, offDP, X, offx, incx, Y,
                                  offy, incy, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            default:
                   return clblasInvalidValue;
            }

    }



//axpy calls
clblasStatus
	clMath::clblas::axpy(
		size_t N,
        cl_float alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{
    return clblasSaxpy(N, alpha, X, offBX, incx, Y, offCY, incy, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
	clMath::clblas::axpy(
		size_t N,
        cl_double alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{

    return clblasDaxpy(N, alpha, X, offBX, incx, Y, offCY, incy, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
	clMath::clblas::axpy(
		size_t N,
        FloatComplex alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{

    return clblasCaxpy(N, alpha, X, offBX, incx, Y, offCY, incy, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
	clMath::clblas::axpy(
		size_t N,
        DoubleComplex alpha,
		cl_mem X,
		size_t offBX,
		int incx,
		cl_mem Y,
		size_t offCY,
		int incy,
		cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
{

    return clblasZaxpy(N, alpha, X, offBX, incx, Y, offCY, incy, numCommandQueues,
                        commandQueues, numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clMath::clblas::rotg(
        DataType type,
        cl_mem SA,
        size_t offSA,
        cl_mem SB,
        size_t offSB,
        cl_mem C,
        size_t offC,
        cl_mem S,
        size_t offS,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
        switch(type)
        {

            case TYPE_FLOAT:
                return clblasSrotg(
                    SA, offSA, SB, offSB, C, offC, S, offS,
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDrotg(
                    SA, offSA, SB, offSB, C, offC, S, offS,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
                return clblasCrotg(
                    SA, offSA, SB, offSB, C, offC, S, offS,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                     eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
                return clblasZrotg(
                    SA, offSA, SB, offSB, C, offC, S, offS,
                    numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            default:
                return  clblasInvalidValue;
        }
    }

clblasStatus
clMath::clblas::rotm(
        DataType type,
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        cl_mem PARAM,
        size_t offParam,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
        switch(type)
        {

            case TYPE_FLOAT:
                return clblasSrotm(
                    N, X, offx, incx, Y, offy, incy, PARAM, offParam,
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDrotm(

                    N, X, offx, incx, Y, offy, incy, PARAM, offParam,
					numCommandQueues, commandQueues, numEventsInWaitList,
                    eventWaitList, events);
            default:
                return  clblasInvalidValue;
        }
    }


clblasStatus
clMath::clblas::rotmg(
        DataType type,
        cl_mem D1,
        size_t offD1,
        cl_mem D2,
        size_t offD2,
        cl_mem X1,
        size_t offX1,
        cl_mem Y1,
        size_t offY1,
        cl_mem PARAM,
        size_t offParam,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
        switch(type)
        {

            case TYPE_FLOAT:
                return clblasSrotmg(
                    D1, offD1, D2, offD2, X1, offX1, Y1, offY1,
                    PARAM, offParam, numCommandQueues, commandQueues,
                    numEventsInWaitList, eventWaitList, events);

            case TYPE_DOUBLE:
                return clblasDrotmg(
                    D1, offD1, D2, offD2, X1, offX1, Y1, offY1,
                    PARAM, offParam, numCommandQueues, commandQueues,
                    numEventsInWaitList, eventWaitList, events);

            default:
                return  clblasInvalidValue;
        }
    }

//ROT
clblasStatus
clMath::clblas::rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
		float C,
		float S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
                return clblasSrot(
                    N, X, offx, incx, Y, offy, incy, (C), (S),
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);
    }



clblasStatus
clMath::clblas::rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        double C,
        double S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
                return clblasDrot(
                    N, X, offx, incx, Y, offy, incy, (C), (S),
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);
    }

clblasStatus
clMath::clblas::rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        FloatComplex C,
        FloatComplex S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
                return clblasCsrot(
                    N, X, offx, incx, Y, offy, incy, CREAL(C), CREAL(S),
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);
    }

clblasStatus
clMath::clblas::rot(
        size_t N,
        cl_mem X,
        size_t offx,
        int incx,
        cl_mem Y,
        size_t offy,
        int incy,
        DoubleComplex C,
        DoubleComplex S,
        cl_uint numCommandQueues,
        cl_command_queue *commandQueues,
        cl_uint numEventsInWaitList,
        const cl_event *eventWaitList,
        cl_event *events)
    {
                return clblasZdrot(
                    N, X, offx, incx, Y, offy, incy, CREAL(C), CREAL(S),
                    numCommandQueues,commandQueues, numEventsInWaitList,
                    eventWaitList, events);
    }


// iAMAX
clblasStatus
clMath::clblas::iamax(
    DataType type,
    size_t N,
    cl_mem iMax,
    size_t offiMax,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuf,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)

    {
            switch(type){

            case TYPE_FLOAT:
            return clblasiSamax( N, iMax, offiMax, X, offx, incx, scratchBuf,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_DOUBLE:
            return clblasiDamax( N, iMax, offiMax, X, offx, incx, scratchBuf,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
            return clblasiCamax( N, iMax, offiMax, X, offx, incx, scratchBuf,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
            return clblasiZamax( N, iMax, offiMax, X, offx, incx, scratchBuf,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            default:
                   return clblasInvalidValue;
            }
    }


clblasStatus
clMath::clblas::nrm2(
    DataType type,
    size_t N,
    cl_mem NRM2,
    size_t offNRM2,
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
            switch(type){

            case TYPE_FLOAT:
            return clblasSnrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_DOUBLE:
            return clblasDnrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            case TYPE_COMPLEX_FLOAT:
            return clblasScnrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                            numCommandQueues,commandQueues, numEventsInWaitList,
                            eventWaitList, events);

            case TYPE_COMPLEX_DOUBLE:
            return clblasDznrm2( N, NRM2, offNRM2, X, offx, incx, scratchBuff,
                                  numCommandQueues,commandQueues, numEventsInWaitList,
                                  eventWaitList, events);

            default:
                   return clblasInvalidValue;
            }

    }
