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
#include <symm.h>
#include<cltypes.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objB, cl_mem objC)
{
    if(objA != NULL)
 	{
    clReleaseMemObject(objA);
	}
	if(objB != NULL)
    {
    clReleaseMemObject(objB);
	}
	if(objC != NULL)
	{
    clReleaseMemObject(objC);
}
}

template <typename T> static void
deleteBuffers(T *A, T *B, T *C, T *backC)
{
    if(A != NULL)
    {
    delete[] A;
    }
	if(B != NULL)
	{
    delete[] B;
	}
	if(C != NULL)
	{
    delete[] C;
}
	if(backC != NULL)
	{
		delete[] backC;
	}
}

template <typename T>
void
symmCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *B, *C, *backC;
	T alpha_, beta_;
    cl_mem bufA, bufB, bufC;
    clMath::BlasBase *base;
    cl_event *events;
	size_t ka, kbc;

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
    if (events == NULL)
    {
    }
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

	if( params->side == clblasLeft )
                ka = params->M;
        else    ka = params->N;

    if( params->order == clblasColumnMajor )
                kbc = params->N;
        else    kbc = params->M;

    size_t lengthA = ka  * params->lda;
    size_t lengthB = kbc * params->ldb;
    size_t lengthC = kbc * params->ldc;

    alpha_ = convertMultiplier<T>(params->alpha);
    beta_ = convertMultiplier<T>(params->beta);

    A 		= new T[ lengthA + params->offa ];
    B   	= new T[ lengthB + params->offb ];
    C   	= new T[ lengthC + params->offc ];
    backC   = new T[ lengthC + params->offc ];

	if((A == NULL) || (B == NULL) || (C == NULL) || (backC == NULL))
	{
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
		deleteBuffers<T>(A, B, C, backC);
        delete[] events;
        SUCCEED();
        return;
	}
    srand(params->seed);
    ::std::cerr << "Generating input data... ";

    int creationFlags = 0, AcreationFlags;
    creationFlags =  creationFlags | RANDOM_INIT;
    creationFlags = ( (params-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    AcreationFlags = ( (params-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_SYMM;

#ifdef __TEST_CSYMM_ACML_NANBUG__
	//
	// NOTE: Whether this clearing to zero is present or not
	// 		 ACML returns "nan" for few csymm cases. This is here
	//		 to make things easier and rule of out-of-bound inputs
	//
	memset(A, 0, (lengthA + params->offa)*sizeof(T));
	memset(B, 0, (lengthB + params->offb)*sizeof(T));
	memset(C, 0, (lengthC + params->offc)*sizeof(T));
#else
	populate( A + params->offa , ka, ka, params-> lda, BlasFn, AcreationFlags);
    populate( B + params->offb , params-> M, params-> N, params-> ldb, BlasFn, creationFlags);
    populate( C + params->offc , params-> M, params-> N, params-> ldc, BlasFn, creationFlags);
#endif

    // Copy C to backX
    memcpy(backC, C, (lengthC + params->offc) * sizeof(T));

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offa) * sizeof(T), 0, CL_MEM_READ_ONLY);
    bufB = base->createEnqueueBuffer(B, (lengthB + params->offb) * sizeof(T), 0, CL_MEM_READ_ONLY);
    bufC = base->createEnqueueBuffer(backC, (lengthC + params->offc) * sizeof(T), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Done" << ::std::endl;
    ::std::cerr << "Calling reference xSYMM routine... ";

	clblasOrder fOrder;
    clblasUplo fUplo;
    clblasSide fSide;
    size_t fN, fM;

	fOrder = params->order;
    fUplo = params->uplo;
    fSide = params->side;
	fM = params->M;
    fN = params->N;

	if (fOrder != clblasColumnMajor) {

           fOrder = clblasColumnMajor;
           fM = params->N;
           fN = params->M;
           fSide = (params->side == clblasLeft)? clblasRight: clblasLeft;
           fUplo = (params->uplo == clblasUpper)? clblasLower: clblasUpper;
       }

	// Call reference blas routine
	clMath::blas::symm(fOrder, fSide, fUplo, fM, fN, alpha_,
                            A, params->offa, params->lda, B, params->offb, params->ldb, beta_, C, params->offc, params->ldc);
	::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufB == NULL) || (bufC == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, C, backC);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xSYMM routine... ";

    err = (cl_int)::clMath::clblas::symm( params->order, params->side, params->uplo, params->M, params->N, alpha_,
                            bufA, params->offa, params->lda, bufB, params->offb, params->ldb, beta_, bufC, params->offc, params->ldc,
							params->numCommandQueues, base->commandQueues(), 0, NULL, events );

    if (err != CL_SUCCESS) {

		releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, C, backC);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SYMM() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {

        releaseMemObjects(bufA, bufB, bufC);
		deleteBuffers<T>(A, B, C, backC);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufC, CL_TRUE, 0,
        (lengthC + params->offc) * sizeof(T), backC, 0,
        NULL, NULL);

    releaseMemObjects(bufA, bufB, bufC);

    // handle lda correctly based on row-major/col-major..
    compareMatrices<T>(params->order, params->M , params->N, (C + params->offc), (backC + params->offc), params->ldc);
    deleteBuffers<T>(A, B, C, backC);
    delete[] events;
}

// Instantiate the test

#ifndef __TEST_CSYMM_ACML_NANBUG__
TEST_P(SYMM, ssymm) {
    TestParams params;

    getParams(&params);
    symmCorrectnessTest<cl_float>(&params);
}

TEST_P(SYMM, dsymm) {
    TestParams params;

    getParams(&params);
    symmCorrectnessTest<cl_double>(&params);
}

TEST_P(SYMM, csymm) {
    TestParams params;

    getParams(&params);
    symmCorrectnessTest<FloatComplex>(&params);
}

TEST_P(SYMM, zsymm) {
    TestParams params;

    getParams(&params);
    symmCorrectnessTest<DoubleComplex>(&params);
}
#else
TEST_P(SYMM, csymm) {
    TestParams params;

    getParams(&params);
    symmCorrectnessTest<FloatComplex>(&params);
}

#endif

