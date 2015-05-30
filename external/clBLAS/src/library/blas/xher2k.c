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

//#define DEBUG_HER2K

extern clblasStatus executeGEMM( CLBlasKargs *kargs, cl_uint numCommandQueues, cl_command_queue *commandQueues, cl_uint numEventsInWaitList,
                                    const cl_event *eventWaitList, cl_event *events);

clblasStatus
doHer2k(
    CLBlasKargs *kargs,
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose transA,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_mem C,
    size_t offc,
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
    cl_event firstHerkCall;
    clblasStatus retCode = clblasSuccess;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    if (numCommandQueues == 0 || commandQueues == NULL) {
        return clblasInvalidValue;
    }
    numCommandQueues = 1;

    if ((numEventsInWaitList !=0) && (eventWaitList == NULL))
    {
        return clblasInvalidEventWaitList;
    }

    // Validate arguments
    if ((retCode = checkMemObjects(A, B, C, true, A_MAT_ERRSET, B_MAT_ERRSET, C_MAT_ERRSET))) {
        return retCode;
    }

    if (transA == clblasTrans) {
        return clblasInvalidValue;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, transA, N, K, A, offa, lda, A_MAT_ERRSET))) {
        return retCode;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, transA, N, K, B, offb, ldb, B_MAT_ERRSET))) {
        return retCode;
    }

    if ((retCode = checkMatrixSizes(kargs->dtype, order, clblasNoTrans, N, N, C, offc, ldc, C_MAT_ERRSET))) {
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
    kargs->offA = offa;
    kargs->offa = offa;
    kargs->lda.matrix = lda;
    kargs->B = B;
    kargs->offBX = offb;
    kargs->ldb.matrix = ldb;
    kargs->C = C;
    kargs->offCY = offc;
    kargs->ldc.matrix = ldc;
    kargs->pigFuncID = CLBLAS_HERK;

    err = executeGEMM(kargs,  numCommandQueues, commandQueues,
                            numEventsInWaitList, eventWaitList, &firstHerkCall);

    if( err == CL_SUCCESS )
    {
        kargs->A = B;
        kargs->offA = offb;
        kargs->offa = offb;
        kargs->lda.matrix = ldb;
        kargs->B = A;
        kargs->offBX = offa;
        kargs->ldb.matrix = lda;

        if( kargs->dtype == TYPE_COMPLEX_FLOAT )
        {
            CIMAG( kargs->alpha.argFloatComplex ) *= -1.0;
            CREAL( kargs->beta.argFloatComplex ) = 1.0;
            CIMAG( kargs->beta.argFloatComplex ) = 0.0;
        }
        else
        {
            CIMAG( kargs->alpha.argDoubleComplex ) *= -1.0;
            CREAL( kargs->beta.argDoubleComplex ) = 1.0;
            CIMAG( kargs->beta.argDoubleComplex ) = 0.0;
        }

        err = executeGEMM(kargs,  numCommandQueues, commandQueues, 1, &firstHerkCall, events);
    }

    return (clblasStatus)err;
}

clblasStatus
clblasCher2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    size_t N,
    size_t K,
    FloatComplex alpha,
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
    CLBlasKargs kargs;
    FloatComplex fBeta;

    memset(&kargs, 0, sizeof(kargs));

    CREAL(fBeta)  = beta;
    CIMAG(fBeta)  = 0.0f;

    kargs.alpha.argFloatComplex = alpha;
    kargs.beta.argFloatComplex = fBeta;
    kargs.dtype = TYPE_COMPLEX_FLOAT;

    if( order == clblasRowMajor )
    {
        CIMAG( kargs.alpha.argFloatComplex ) *= -1.0;
    }

    return doHer2k(&kargs, order, uplo, trans, N, K, A, offa, lda, B, offb, ldb,
                    C, offc, ldc, numCommandQueues, commandQueues,
                    numEventsInWaitList, eventWaitList, events);
}

clblasStatus
clblasZher2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    size_t N,
    size_t K,
    DoubleComplex alpha,
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
    CLBlasKargs kargs;
    DoubleComplex fBeta;

    memset(&kargs, 0, sizeof(kargs));

    CREAL(fBeta)  = beta;
    CIMAG(fBeta)  = 0.0f;

    kargs.alpha.argDoubleComplex = alpha;
    kargs.beta.argDoubleComplex = fBeta;

    kargs.dtype = TYPE_COMPLEX_DOUBLE;

    if( order == clblasRowMajor )
    {
        CIMAG( kargs.alpha.argDoubleComplex ) *= -1.0;
    }

    return doHer2k(&kargs, order, uplo, trans, N, K, A, offa, lda, B, offb, ldb,
                    C, offc, ldc, numCommandQueues, commandQueues,
                    numEventsInWaitList, eventWaitList, events);
}

