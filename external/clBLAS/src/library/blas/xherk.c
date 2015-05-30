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

extern clblasStatus executeGEMM( CLBlasKargs *kargs, cl_uint numCommandQueues, cl_command_queue *commandQueues, cl_uint numEventsInWaitList,
                                    const cl_event *eventWaitList, cl_event *events);

clblasStatus
doHerk(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
{
    clblasStatus err;
    clblasUplo fUplo;
    clblasTranspose fTransA;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    if (numCommandQueues == 0 || commandQueues == NULL) {
        return clblasInvalidValue;
    }

    if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
    {
        return clblasInvalidEventWaitList;
    }

    // Validate arguments
    if ((retCode = checkMemObjects(A, C, NULL, false, A_MAT_ERRSET, C_MAT_ERRSET, END_ERRSET))) {
        return retCode;
    }

    if (transA == clblasTrans) {
        return clblasInvalidValue;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, transA, N, K, A, offA, lda, A_MAT_ERRSET))) {
        return retCode;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, false, N, N, C, offC, ldc, C_MAT_ERRSET))) {
        return retCode;
    }

    if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
    {
        return clblasInvalidEventWaitList;
    }

    fUplo = (order == clblasRowMajor) ? ((uplo == clblasLower) ? clblasUpper : clblasLower) : uplo;
    fTransA = (order == clblasRowMajor) ? ((transA == clblasNoTrans) ? clblasConjTrans : clblasNoTrans) : transA;
    kargs->order = (order == clblasRowMajor) ? clblasColumnMajor : order;
    kargs->transA = fTransA;
    kargs->transB = (fTransA == clblasNoTrans) ? clblasConjTrans : clblasNoTrans;
    kargs->uplo = fUplo;
    kargs->M = N;
    kargs->N = N;
    kargs->K = K;
    kargs->A = A;
    kargs->offA = offA;
    kargs->offa = offA;
    kargs->lda.matrix = lda;
    kargs->B = A;
    kargs->offBX = offA;
    kargs->ldb.matrix = lda;
    kargs->C = C;
    kargs->offCY = offC;
    kargs->ldc.matrix = ldc;
    kargs->pigFuncID = CLBLAS_HERK;

	err = CL_SUCCESS;
	#ifdef DEBUG_HERK
        printf("doHerk called\n");
    #endif

    numCommandQueues = 1;
    // Call GEMM to handle HERK.
    err = executeGEMM(kargs,  numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events);
/*
    listInitHead(&seq);
    err = makeSolutionSeq(CLBLAS_GEMM, kargs, numCommandQueues, commandQueues,
        numEventsInWaitList, eventWaitList, events, &seq);
    if (err == CL_SUCCESS) {
        err = executeSolutionSeq(&seq);
    }

    freeSolutionSeq(&seq);
*/
    return (clblasStatus)err;
}

clblasStatus
clblasCherk(
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
    CLBlasKargs kargs;
    FloatComplex fAlpha, fBeta;

    memset(&kargs, 0, sizeof(kargs));

    CREAL(fAlpha) = alpha;
    CIMAG(fAlpha) = 0.0f;
    CREAL(fBeta)  = beta;
    CIMAG(fBeta)  = 0.0f;

    kargs.alpha.argFloatComplex = fAlpha;
    kargs.beta.argFloatComplex = fBeta;
    kargs.dtype = TYPE_COMPLEX_FLOAT;

    return doHerk(&kargs, order, uplo, transA, N, K, A, offA, lda,
                    C, offC, ldc, numCommandQueues, commandQueues,
                    numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZherk(
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
    CLBlasKargs kargs;
    DoubleComplex fAlpha, fBeta;

    memset(&kargs, 0, sizeof(kargs));

    CREAL(fAlpha) = alpha;
    CIMAG(fAlpha) = 0.0f;
    CREAL(fBeta)  = beta;
    CIMAG(fBeta)  = 0.0f;

    kargs.alpha.argDoubleComplex = fAlpha;
    kargs.beta.argDoubleComplex = fBeta;

    kargs.dtype = TYPE_COMPLEX_DOUBLE;

    return doHerk(&kargs, order, uplo, transA, N, K, A, offA, lda,
                    C, offC, ldc, numCommandQueues, commandQueues,
                    numEventsInWaitList, eventWaitList, events);
}

