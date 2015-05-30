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
#include <her2k.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objC, cl_mem objB)
{
    if(objA != NULL)
		clReleaseMemObject(objA);
	if(objC != NULL)
   		clReleaseMemObject(objC);
    if(objB != NULL)
   		clReleaseMemObject(objB);
}

template <typename T> static void
deleteBuffers(T *A, T *B, T *blasC, T *clblasC)
{
	if(A != NULL)
    	delete[] A;
    if(B != NULL)
    	delete[] B;
	if(blasC != NULL)
    	delete[] blasC;
	if(clblasC != NULL)
    	delete[] clblasC;
}

template <typename T>
void
her2kCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *B, *blasC, *clblasC;
    T alpha, beta;
    cl_mem bufA, bufC, bufB;
    clMath::BlasBase *base;
    cl_event *events;

    if (params->transA == clblasTrans) {
        ::std::cerr << ">> her2k(TRANSPOSE) for complex numbers "
                           "is not allowed." << ::std::endl <<
                           ">> Test skipped." << ::std::endl;
            SUCCEED();
            return;
        }

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

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    A = new T[params->rowsA * params->columnsA];
    B = new T[params->rowsB * params->columnsB];
    blasC = new T[params->rowsC * params->columnsC];
    clblasC = new T[params->rowsC * params->columnsC];

	if((A == NULL) || (B == NULL) || (blasC == NULL) || (clblasC == NULL))
	{
		deleteBuffers<T>(A, B, blasC, clblasC);
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        delete[] events;
        SUCCEED();
        return;
	}

    srand(params->seed);

    alpha = convertMultiplier<T>(params->alpha);
    beta  = convertMultiplier<T>(params->beta);

    ::std::cerr << "Generating input data... ";

    clblasTranspose ftransB = (params->transA==clblasNoTrans)? clblasConjTrans: clblasNoTrans;

    randomGemmMatrices<T>(params->order, params->transA, ftransB,
                                params->N, params->N, params->K, true, &alpha, A, params->lda,
                                B, params->ldb, true, &beta, blasC, params->ldc);

    memcpy(clblasC, blasC, params->rowsC * params->columnsC * sizeof(*blasC));
    ::std::cerr << "Done" << ::std::endl;

    bufA = base->createEnqueueBuffer(A, params->rowsA * params->columnsA * sizeof(*A), params->offA * sizeof(*A),
                                     CL_MEM_READ_ONLY);
    bufB = base->createEnqueueBuffer(B, params->rowsB * params->columnsB * sizeof(*B), params->offBX * sizeof(*B),
                                     CL_MEM_READ_ONLY);
    bufC = base->createEnqueueBuffer(clblasC, params->rowsC * params->columnsC * sizeof(*clblasC),
                                     params->offCY * sizeof(*clblasC),
                                     CL_MEM_READ_WRITE);

    if ((bufA == NULL) || (bufB == NULL)|| (bufC == NULL)) {
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

    ::std::cerr << "Calling reference xHER2K routine... ";
    T fAlpha = alpha;
    if (params->order == clblasColumnMajor) {
        ::clMath::blas::her2k(clblasColumnMajor, params->uplo, params->transA,
                                params->N, params->K, fAlpha, A, 0, params->lda, B, 0, params->ldb,
                                CREAL(beta), blasC, 0, params->ldc);
    }
    else {

		CIMAG( fAlpha ) *= -1.0;        // According to netlib C- interface
        clblasTranspose fTransA = (params->transA == clblasNoTrans) ? clblasConjTrans : clblasNoTrans;
		clblasUplo      fUplo   = (params->uplo == clblasUpper) ? clblasLower : clblasUpper;

		::clMath::blas::her2k(clblasColumnMajor, fUplo, fTransA, params->N, params->K, fAlpha,
						        A, 0, params->lda, B, 0, params->ldb, CREAL(beta), blasC, 0, params->ldc);

    }
    ::std::cerr << "Done" << ::std::endl;

    ::std::cerr << "Calling clblas xHER2K routine... ";
    err = (cl_int)::clMath::clblas::her2k(params->order, params->uplo,
                                         params->transA, params->N, params->K,
                                         alpha, bufA, params->offA, params->lda, bufB, params->offBX, params->ldb,
                                         CREAL(beta), bufC, params->offCY,
                                         params->ldc, params->numCommandQueues,
                                         base->commandQueues(), 0, NULL,
                                         events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, blasC, clblasC);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::HER2K() failed";
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

    clEnqueueReadBuffer(base->commandQueues()[0], bufC, CL_TRUE, params->offCY * sizeof(*clblasC),
                        params->rowsC * params->columnsC * sizeof(*clblasC), clblasC, 0, NULL, NULL);

    releaseMemObjects(bufA, bufB, bufC);
    compareMatrices<T>(params->order, params->N, params->N, blasC, clblasC, params->ldc);

    deleteBuffers<T>(A, B, blasC, clblasC);
    delete[] events;
}

// Instantiate the test

TEST_P(HER2K, cher2k) {
    TestParams params;

    getParams(&params);
    her2kCorrectnessTest<FloatComplex>(&params);
}

TEST_P(HER2K, zher2k) {
    TestParams params;

    getParams(&params);
    her2kCorrectnessTest<DoubleComplex>(&params);
}
