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


#include <stdlib.h>             // srand()
#include <string.h>             // memcpy()
#include <gtest/gtest.h>
#include <clBLAS.h>

#include <common.h>
#include <blas-internal.h>
#include <blas-wrapper.h>
#include <clBLAS-wrapper.h>
#include <BlasBase.h>
#include <blas-random.h>
#include <trmm.h>

#include "tcase-filter.h"

static void
releaseMemObjects(cl_mem A, cl_mem B)
{
    clReleaseMemObject(A);
    clReleaseMemObject(B);
}

template <typename T> static void
deleteBuffers(T *A, T *blasB, T *clblasB)
{
    delete[] A;
    delete[] blasB;
    delete[] clblasB;
}

template <typename T>
void
trmmCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *blasB, *clblasB;
    T alpha;
    cl_mem bufA, bufB;
    clMath::BlasBase *base;
    bool useAlpha;
    cl_event *events;
    bool isComplex;

    base = clMath::BlasBase::getInstance();
    if ((typeid(T) == typeid(cl_double) ||
         typeid(T) == typeid(DoubleComplex)) &&
        !base->isDevSupportDoublePrecision()) {

        std::cerr << ">> WARNING: The target device doesn't support native "
                     "double precision floating point arithmetic" <<
                     std::endl << ">> Test skipped" << std::endl;
        SUCCEED();
        return;
    }

    isComplex = ((typeid(T) == typeid(FloatComplex)) ||
                 (typeid(T) == typeid(DoubleComplex)));
    if (canCaseBeSkipped(params, isComplex)) {
        std::cerr << ">> Test is skipped because it has no importance for this "
                     "level of coverage" << std::endl;
        SUCCEED();
        return;
    }

    useAlpha = base->useAlpha();

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    A = new T[params->rowsA * params->columnsA];
    blasB = new T[params->rowsB * params->columnsB];
    clblasB = new T[params->rowsB * params->columnsB];
    alpha = ZERO<T>();

    srand(params->seed);
    if (useAlpha) {
        alpha = convertMultiplier<T>(params->alpha);
    }

    ::std::cerr << "Generating input data... ";
    randomTrmmMatrices<T>(params->order, params->side, params->uplo,
        params->diag, params->M, params->N, useAlpha,
        &alpha, A, params->lda, blasB, params->ldb);
    memcpy(clblasB, blasB, params->rowsB * params->columnsB * sizeof(*blasB));
    ::std::cerr << "Done" << ::std::endl;

    ::std::cerr << "Calling reference xTRMM routine... ";
    if (params->order == clblasColumnMajor) {
        ::clMath::blas::trmm(clblasColumnMajor, params->side, params->uplo,
            params->transA, params->diag, params->M, params->N, alpha,
            A, params->lda, blasB, params->ldb);
    }
    else {
        T *reorderedA = new T[params->rowsA * params->columnsA];
        T *reorderedB = new T[params->rowsB * params->columnsB];

        reorderMatrix<T>(clblasRowMajor, params->rowsA, params->columnsA,
                         A, reorderedA);
        reorderMatrix<T>(clblasRowMajor, params->rowsB, params->columnsB,
                         blasB, reorderedB);
        ::clMath::blas::trmm(clblasColumnMajor, params->side, params->uplo,
            params->transA, params->diag, params->M, params->N, alpha,
            reorderedA, params->rowsA, reorderedB, params->rowsB);
        reorderMatrix<T>(clblasColumnMajor, params->rowsB, params->columnsB,
                         reorderedB, blasB);

        delete[] reorderedB;
        delete[] reorderedA;
    }
    ::std::cerr << "Done" << ::std::endl;

    bufA = base->createEnqueueBuffer(A, params->rowsA * params->columnsA *
                                     sizeof(*A), params->offA * sizeof(*A),
                                     CL_MEM_READ_ONLY);
    bufB = base->createEnqueueBuffer(clblasB, params->rowsB * params->columnsB *
                                     sizeof(*clblasB),
                                     params->offBX * sizeof(*clblasB),
                                     CL_MEM_READ_WRITE);
    if ((bufA == NULL) || (bufB == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, blasB, clblasB);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTRMM routine... ";
    err = (cl_int)::clMath::clblas::trmm(params->order, params->side,
        params->uplo, params->transA, params->diag, params->M, params->N,
        alpha, bufA, params->offA, params->lda, bufB, params->offBX,
        params->ldb, params->numCommandQueues, base->commandQueues(),
        0, NULL, events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, blasB, clblasB);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TRMM() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, blasB, clblasB);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufB, CL_TRUE,
                        params->offBX * sizeof(*clblasB),
                        params->rowsB * params->columnsB * sizeof(*clblasB),
                        clblasB, 0, NULL, NULL);

    releaseMemObjects(bufA, bufB);
    compareMatrices<T>(params->order, params->M, params->N, blasB, clblasB,
                       params->ldb);
    deleteBuffers<T>(A, blasB, clblasB);
    delete[] events;
}

// Instantiate the test

TEST_P(TRMM, strmm) {
    TestParams params;

    getParams(&params);
    trmmCorrectnessTest<cl_float>(&params);
}

TEST_P(TRMM, dtrmm) {
    TestParams params;

    getParams(&params);
    trmmCorrectnessTest<cl_double>(&params);
}

TEST_P(TRMM, ctrmm) {
    TestParams params;

    getParams(&params);
    trmmCorrectnessTest<FloatComplex>(&params);
}

TEST_P(TRMM, ztrmm) {
    TestParams params;

    getParams(&params);
    trmmCorrectnessTest<DoubleComplex>(&params);
}
