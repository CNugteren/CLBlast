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
#include <symv.h>

#include "tcase-filter.h"

static void
releaseMemObjects(cl_mem objA, cl_mem objX, cl_mem objY)
{
    clReleaseMemObject(objA);
    clReleaseMemObject(objX);
    clReleaseMemObject(objY);
}

template <typename T> static void
deleteBuffers(T *A, T *X, T *blasY, T *clblasY)
{
    delete[] A;
    delete[] X;
    delete[] blasY;
    delete[] clblasY;
}

template <typename T>
void
symvCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *B, *blasC, *clblasC, *X, *Y;
    T alpha, beta;
    cl_mem bufA, bufB, bufC;
    clMath::BlasBase *base;
    bool useAlpha, useBeta;
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
    useBeta = base->useBeta();
    alpha = ZERO<T>();
    beta = ZERO<T>();

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    A = new T[params->rowsA * params->columnsA];
    // X and Y are rows or columns in matrixes B and C
    B = new T[params->rowsB * params->columnsB];
    blasC = new T[params->rowsC * params->columnsC];
    clblasC = new T[params->rowsC * params->columnsC];
    X = &B[params->offBX];
    Y = &blasC[params->offCY];

    srand(params->seed);
    if (useAlpha) {
        alpha = convertMultiplier<T>(params->alpha);
    }
    if (useBeta) {
        beta = convertMultiplier<T>(params->beta);
    }

    ::std::cerr << "Generating input data... ";
    setNans<T>(params->rowsA * params->columnsA, A);
    setNans<T>(params->rowsB * params->columnsB, B);
    setNans<T>(params->rowsC * params->columnsC, blasC);
    randomGemmMatrices(params->order, clblasNoTrans, clblasNoTrans,
                       params->N, params->N, params->N, useAlpha, &alpha, A,
                       params->lda, B, params->ldb, useBeta, &beta, blasC,
                       params->ldc);
    // set to NAN elements which must not be accessed
    // in matrix A
    setTriangleNans<T>(params->order, params->uplo, params->N, A, params->lda);

    // in matrix B containing vector X
    setVectorNans<T>(params->offBX, abs(params->incx), B, params->N,
                  params->columnsB * params->rowsB);
    // in matrix C containing vector Y
    setVectorNans<T>(params->offCY, abs(params->incy), blasC, params->N,
                  params->columnsC * params->rowsC);
    memcpy(clblasC, blasC, params->rowsC * params->columnsC * sizeof(*clblasC));
    ::std::cerr << "Done" << ::std::endl;

    ::std::cerr << "Calling reference xSYMV routine... ";

    if (params->order == clblasColumnMajor) {
        ::clMath::blas::symv(clblasColumnMajor, params->uplo,
                          params->N, alpha, A, params->lda,
                          X, params->incx, beta, Y, params->incy);
    }
    else {
        T *reorderedA = new T[params->rowsA * params->columnsA];

        reorderMatrix<T>(clblasRowMajor, params->rowsA, params->columnsA,
                         A, reorderedA);
        ::clMath::blas::symv(clblasColumnMajor, params->uplo,
                          params->N, alpha, reorderedA, params->rowsA,
                          X, params->incx, beta, Y, params->incy);

        delete[] reorderedA;
    }
    ::std::cerr << "Done" << ::std::endl;

    bufA = base->createEnqueueBuffer(A, params->rowsA * params->columnsA *
                                     sizeof(*A), params->offA * sizeof(*A),
                                     CL_MEM_READ_ONLY);
    bufB = base->createEnqueueBuffer(B, params->rowsB * params->columnsB *
                                     sizeof(*X), 0, CL_MEM_READ_ONLY);
    bufC = base->createEnqueueBuffer(clblasC, params->rowsC * params->columnsC *
                                     sizeof(*clblasC), 0, CL_MEM_READ_WRITE);
    if ((bufA == NULL) || (bufB == NULL) || (bufC == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, blasC, clblasC);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xSYMV routine... ";
    err = (cl_int)::clMath::clblas::symv(params->order, params->uplo,
        params->N, alpha, bufA, params->offA, params->lda, bufB, params->offBX,
        params->incx, beta, bufC, params->offCY, params->incy,
        params->numCommandQueues, base->commandQueues(),
        0, NULL, events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, blasC, clblasC);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SYMV() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, blasC, clblasC);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufC, CL_TRUE, 0,
                        params->rowsC * params->columnsC * sizeof(*clblasC),
                        clblasC, 0, NULL, NULL);

    releaseMemObjects(bufA, bufB, bufC);

    compareVectors(params->offCY, params->N, abs(params->incy),
                   params->columnsC * params->rowsC, blasC, clblasC);

    deleteBuffers<T>(A, B, blasC, clblasC);
    delete[] events;
}

// Instantiate the test

TEST_P(SYMV, ssymv) {
    TestParams params;

    getParams(&params);
    symvCorrectnessTest<cl_float>(&params);
}

TEST_P(SYMV, dsymv) {
    TestParams params;

    getParams(&params);
    symvCorrectnessTest<cl_double>(&params);
}
