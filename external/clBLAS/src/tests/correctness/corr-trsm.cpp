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
#include <trsm.h>

#include "trsm-delta.h"
#include "tcase-filter.h"

static void
releaseMemObjects(cl_mem A, cl_mem B)
{
    clReleaseMemObject(A);
    clReleaseMemObject(B);
}

template <typename T> static void
deleteBuffers(T *A, T *B, T *blasB, T *clblasB, cl_double *delta)
{
    delete[] A;
    delete[] B;
    delete[] blasB;
    delete[] clblasB;
    delete[] delta;
}

template <typename T>
void
trsmCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *B, *blasB, *clblasB;
    T alpha;
    cl_mem bufA, bufB;
    cl_double *delta;
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
    alpha = ZERO<T>();

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    A = new T[params->rowsA * params->columnsA];
    B = new T[params->rowsB * params->columnsB];
    blasB = new T[params->rowsB * params->columnsB];
    clblasB = new T[params->rowsB * params->columnsB];
    delta = new cl_double[params->rowsB * params->columnsB];

    srand(params->seed);
    if (useAlpha) {
        alpha = convertMultiplier<T>(params->alpha);
    }

    ::std::cerr << "Generating input data... ";

    randomTrsmMatrices<T>(params->order, params->side, params->uplo,
        params->diag, params->M, params->N, useAlpha,
        &alpha, A, params->lda, B, params->ldb);

    memcpy(blasB, B, params->rowsB * params->columnsB * sizeof(*B));
    memcpy(clblasB, B, params->rowsB * params->columnsB * sizeof(*B));
    ::std::cerr << "Done" << ::std::endl;

    ::std::cerr << "Calling reference xTRSM routine... ";
    if (params->order == clblasColumnMajor) {
        ::clMath::blas::trsm(clblasColumnMajor, params->side, params->uplo,
            params->transA, params->diag, params->M, params->N, alpha, A,
            params->lda, blasB, params->ldb);
    }
    else {
        T *reorderedA = new T[params->rowsA * params->columnsA];
        T *reorderedB = new T[params->rowsB * params->columnsB];

        reorderMatrix<T>(clblasRowMajor, params->rowsA, params->columnsA,
                         A, reorderedA);
        reorderMatrix<T>(clblasRowMajor, params->rowsB, params->columnsB,
                         blasB, reorderedB);

        ::clMath::blas::trsm(clblasColumnMajor, params->side, params->uplo,
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
        deleteBuffers<T>(A, B, blasB, clblasB, delta);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTRSM routine... ";
    err = (cl_int)::clMath::clblas::trsm(params->order, params->side,
        params->uplo, params->transA, params->diag, params->M, params->N,
        alpha, bufA, params->offA, params->lda, bufB, params->offBX,
        params->ldb, params->numCommandQueues, base->commandQueues(),
        0, NULL, events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, B, blasB, clblasB, delta);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TRSM() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, B, blasB, clblasB, delta);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufB, CL_TRUE,
                        params->offBX * sizeof(*clblasB),
                        params->rowsB * params->columnsB * sizeof(*clblasB),
                        clblasB, 0, NULL, NULL);

    releaseMemObjects(bufA, bufB);

    trsmDelta<T>(params->order, params->side, params->uplo, params->transA,
        params->diag, params->M, params->N, A, params->lda, B, params->ldb,
        alpha, delta);

    compareMatrices<T>(params->order, params->M, params->N, blasB, clblasB,
                       params->ldb, delta);
    deleteBuffers<T>(A, B, blasB, clblasB, delta);
    delete[] events;
}

// Instantiate the test

TEST_P(TRSM, strsm) {
    TestParams params;

    getParams(&params);
    trsmCorrectnessTest<cl_float>(&params);
}

TEST_P(TRSM, dtrsm) {
    TestParams params;

    getParams(&params);
    trsmCorrectnessTest<cl_double>(&params);
}

TEST_P(TRSM, ctrsm) {
    TestParams params;

    getParams(&params);
    trsmCorrectnessTest<FloatComplex>(&params);
}

TEST_P(TRSM, ztrsm) {
    TestParams params;

    getParams(&params);
    trsmCorrectnessTest<DoubleComplex>(&params);
}



// ====================================
// Adding some tests to catch bugs in the scenario where lda != M


int arithsum(int i)
{
	int j;
	for(j=i-1; j>0; j--)
		i += j;
	return i;
}

template <typename T>
void AssignA(T *A, size_t i, size_t j, size_t ld)
{
	A[i*ld + j] = j == i ? (j+1) : ( j > i ? 0 : 1.0 );
}

template <>
void AssignA(FloatComplex *A, size_t i, size_t j, size_t ld)
{
	FloatComplex *Ac = (FloatComplex *)A;
	Ac[i*ld + j].s[0] = j == i ? (j+1) : ( j > i ? 0 : 1.0 );
	Ac[i*ld + j].s[1] = 0;
}

template <>
void AssignA(DoubleComplex *A, size_t i, size_t j, size_t ld)
{
	DoubleComplex *Az = (DoubleComplex *)A;
	Az[i*ld + j].s[0] = j == i ? (j+1) : ( j > i ? 0 : 1.0 );
	Az[i*ld + j].s[1] = 0;
}

template <typename T>
void AssignB(T *B, size_t i, size_t j, size_t ld, size_t M)
{
	B[i*ld + j] = arithsum(M) - arithsum(j+1) + (j+1)*(j+1);
}

template <>
void AssignB(FloatComplex *B, size_t i, size_t j, size_t ld, size_t M)
{
	FloatComplex *Bc = (FloatComplex *)B;
	Bc[i*ld + j].s[0] = arithsum(M) - arithsum(j+1) + (j+1)*(j+1);
	Bc[i*ld + j].s[1] = 0;
}

template <>
void AssignB(DoubleComplex *B, size_t i, size_t j, size_t ld, size_t M)
{
	DoubleComplex *Bz = (DoubleComplex *)B;
	Bz[i*ld + j].s[0] = arithsum(M) - arithsum(j+1) + (j+1)*(j+1);
	Bz[i*ld + j].s[1] = 0;
}

template <typename T>
void local_assert(T x, T y, T d)
{
	ASSERT_NEAR(x, y, d);
}

template <>
void local_assert<FloatComplex>(FloatComplex x, FloatComplex y, FloatComplex d)
{
	ASSERT_NEAR(x.s[0], y.s[0], d.s[0]);
	ASSERT_NEAR(x.s[1], y.s[1], d.s[1]);
}

template <>
void local_assert<DoubleComplex>(DoubleComplex x, DoubleComplex y, DoubleComplex d)
{
	ASSERT_NEAR(x.s[0], y.s[0], d.s[0]);
	ASSERT_NEAR(x.s[1], y.s[1], d.s[1]);
}


template <typename T>
void Extratest(size_t M, size_t N, size_t lda, size_t ldb, T alpha, T delta)
{
	T *A, *B, *blasB, *clblasB;
	cl_mem bufA, bufB;
	clMath::BlasBase *base;
    cl_event *events;
	cl_int err;

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


	clblasOrder order = clblasColumnMajor;
	clblasSide side = clblasLeft;
	clblasUplo uplo = clblasUpper;
	clblasTranspose trans = clblasNoTrans;
	clblasDiag diag = clblasNonUnit;

	A = new T[M * lda];
    B = new T[N * ldb];
    blasB = new T[N * ldb];
    clblasB = new T[N * ldb];

	memset(A, 0, M*lda*sizeof(T));
	memset(B, 0, N*ldb*sizeof(T));

	for(int i=0; i<M; i++) // down each column
	{
		for(int j=0; j<M; j++) // down each row
		{
			AssignA<T>(A, i, j, lda);
		}
	}

	for(int i=0; i<N; i++) // down each column
	{
		for(int j=0; j<M; j++) // down each row
		{
			AssignB<T>(B, i, j, ldb, M);
		}
	}

    memcpy(blasB, B, N*ldb*sizeof(T));
    memcpy(clblasB, B, N*ldb*sizeof(T));

	::std::cerr << "Calling reference xTRSM routine... ";
	::clMath::blas::trsm(order, side, uplo, trans, diag, M, N, alpha, A, lda, blasB, ldb);


    bufA = base->createEnqueueBuffer(A, M*lda*sizeof(T), 0, CL_MEM_READ_ONLY);
    bufB = base->createEnqueueBuffer(clblasB, N*ldb*sizeof(T), 0, CL_MEM_READ_WRITE);

    events = new cl_event[1];
    memset(events, 0, sizeof(cl_event));

    if ((bufA == NULL) || (bufB == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, B, blasB, clblasB, NULL);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTRSM routine... ";
    err = (cl_int)::clMath::clblas::trsm(order, side, uplo, trans, diag, M, N, alpha, bufA, 0, lda, bufB, 0, ldb,
				1, base->commandQueues(), 0, NULL, events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, B, blasB, clblasB, NULL);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TRSM() failed";
    }

    err = waitForSuccessfulFinish(1, base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB);
        deleteBuffers<T>(A, B, blasB, clblasB, NULL);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufB, CL_TRUE,
                        0, N*ldb*sizeof(T), clblasB, 0, NULL, NULL);

    releaseMemObjects(bufA, bufB);

	// Validate the answer
	for(int i=0; i<N; i++) // down each column
	{
		for(int j=0; j<ldb; j++) // down each row
		{
			local_assert(blasB[i*ldb + j], clblasB[i*ldb + j], delta);
		}
	}

	deleteBuffers<T>(A, B, blasB, clblasB, NULL);
    delete[] events;
}

#define ETST_TOLERENCE 1E-5

TEST(TRSM_extratest, strsm)
{
	Extratest<float>(5, 2, 32, 32, 1.0f, ETST_TOLERENCE);
}

TEST(TRSM_extratest, dtrsm)
{
	Extratest<double>(5, 2, 32, 32, 1.0, ETST_TOLERENCE);
}

TEST(TRSM_extratest, ctrsm)
{
	FloatComplex alpha = floatComplex(1.0f, 0);
	FloatComplex delta = floatComplex(ETST_TOLERENCE, ETST_TOLERENCE);
	Extratest<FloatComplex>(5, 2, 32, 32, alpha, delta);
}

TEST(TRSM_extratest, ztrsm)
{
	DoubleComplex alpha = doubleComplex(1.0, 0);
	DoubleComplex delta = doubleComplex(ETST_TOLERENCE, ETST_TOLERENCE);
	Extratest<DoubleComplex>(5, 2, 32, 32, alpha, delta);
}