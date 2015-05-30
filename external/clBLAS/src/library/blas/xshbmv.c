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
doSHbmv(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
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

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    if ((commandQueues == NULL) || (numCommandQueues == 0))
    {
        return clblasInvalidValue;
    }

    if (commandQueues[0] == NULL)
    {
        return clblasInvalidCommandQueue;
    }

    if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
    {
        return clblasInvalidEventWaitList;
    }
    /* Validate arguments */

    if ((retCode = checkMemObjects(A, x, y, true, A_MAT_ERRSET, X_VEC_ERRSET, Y_VEC_ERRSET)))
    {
        return retCode;
    }

    if ((retCode = checkBandedMatrixSizes(kargs->dtype, order, clblasNoTrans,
                                          N, N, K, 0, A, offa, lda, A_MAT_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, x, offx, incx, X_VEC_ERRSET))) {
        return retCode;
    }
    if ((retCode = checkVectorSizes(kargs->dtype, N, y, offy, incy, Y_VEC_ERRSET))) {
        return retCode;
    }

    /* numCommandQueues will be hardcoded to 1 as of now. No multi-gpu support */
    numCommandQueues = 1;

    kargs->order = order;
    kargs->uplo = uplo;
    kargs->transA = clblasNoTrans;
    kargs->N = N;
    kargs->M = N;
    kargs->KL = K;
    kargs->KU = K;
    kargs->A = A;
    kargs->offA = offa;
    kargs->offa = offa;
    kargs->lda.matrix = lda;
    kargs->B = x;
    kargs->offBX = offx;
    kargs->ldb.vector = incx;
    kargs->C = y;
    kargs->offCY = offy;
    kargs->ldc.vector = incy;

    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_GBMV, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events, &seq);

    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);

    return (clblasStatus)err;
}

clblasStatus
clblasSsbmv(
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
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_FLOAT;
    kargs.pigFuncID = CLBLAS_SBMV;
    kargs.alpha.argFloat = alpha;
    kargs.beta.argFloat = beta;

    return doSHbmv(&kargs, order, uplo, N, K, A, offa, lda, X, offx, incx,
                  Y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasDsbmv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
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

    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_DOUBLE;
    kargs.pigFuncID = CLBLAS_SBMV;
    kargs.alpha.argDouble = alpha;
    kargs.beta.argDouble = beta;

    return doSHbmv(&kargs, order, uplo, N, K, A, offa, lda, X, offx, incx,
                  Y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasChbmv(
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
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_FLOAT;
    kargs.pigFuncID = CLBLAS_HBMV;
    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = beta;

    return doSHbmv(&kargs, order, uplo, N, K, A, offa, lda, X, offx, incx,
                  Y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZhbmv(
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
    CLBlasKargs kargs;

    memset(&kargs, 0, sizeof(kargs));
    kargs.dtype = TYPE_COMPLEX_DOUBLE;
    kargs.pigFuncID = CLBLAS_HBMV;
    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = beta;

    return doSHbmv(&kargs, order, uplo, N, K, A, offa, lda, X, offx, incx,
                  Y, offy, incy, numCommandQueues, commandQueues,
                  numEventsInWaitList, eventWaitList, events);
}
