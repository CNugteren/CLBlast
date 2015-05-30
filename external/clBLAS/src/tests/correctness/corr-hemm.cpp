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
#include <hemm.h>
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
    delete[] backC;// To hold the original C
}
}

template <typename T>
void
hemmCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *B, *C, *backC;
	T alpha_, beta_;
    cl_mem bufA, bufB, bufC;
    clMath::BlasBase *base;
    cl_event *events;
	size_t ka, kbc;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(DoubleComplex)) &&
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
        std::cerr << ">> WARNING: Unable to allocate memory for events"  <<
                     std::endl << ">> Test skipped" << std::endl;
        SUCCEED();
        return;
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


    A 		= new T[ lengthA + params->offA ];
    B   	= new T[ lengthB + params->offBX ];
    C   	= new T[ lengthC + params->offCY ];
    backC   = new T[ lengthC + params->offCY ];

	if((A == NULL) || (B == NULL) || (C == NULL) || (backC == NULL))
    {
        ::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(A, B, C, backC);
        delete[] events;
		SUCCEED();
        return;
    }

    srand(params->seed);

    ::std::cerr << "Generating input data... " << std::endl;

    int creationFlags = 0, AcreationFlags;
    creationFlags =  creationFlags | RANDOM_INIT;
    creationFlags = ( (params-> order) == clblasRowMajor)? (creationFlags | ROW_MAJOR_ORDER) : (creationFlags);
    AcreationFlags = ( (params-> uplo) == clblasLower)? (creationFlags | LOWER_HALF_ONLY) : (creationFlags | UPPER_HALF_ONLY);
	BlasRoutineID BlasFn = CLBLAS_HEMM;

	populate( A + params->offA , ka, ka, params-> lda, BlasFn, AcreationFlags);
    populate( B + params->offBX , params-> M, params-> N, params-> ldb, BlasFn, creationFlags);
    populate( C + params->offCY , params-> M, params-> N, params-> ldc, BlasFn, creationFlags);

	memcpy(backC, C, (lengthC + params->offCY) * sizeof(T));
    //printMatrixBlock( params->order, 0, 0, params->M, params->N, params->ldc, backC);

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offA) * sizeof(T), 0, CL_MEM_READ_ONLY);
    bufB = base->createEnqueueBuffer(B, (lengthB + params->offBX) * sizeof(T), 0, CL_MEM_READ_ONLY);
    bufC = base->createEnqueueBuffer(backC, (lengthC + params->offCY) * sizeof(T), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Done" << ::std::endl;
    ::std::cerr << "Calling reference xHEMM routine... ";

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
    clMath::blas::hemm(fOrder, fSide, fUplo, fM, fN, alpha_,
                            A, params->offA, params->lda, B, params->offBX, params->ldb, beta_, C, params->offCY, params->ldc);
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

    ::std::cerr << "Calling clblas xHEMM routine... ";

    err = (cl_int)::clMath::clblas::hemm( params->order, params->side, params->uplo, params->M, params->N, alpha_,
                            bufA, params->offA, params->lda, bufB, params->offBX, params->ldb, beta_, bufC, params->offCY, params->ldc,
							params->numCommandQueues, base->commandQueues(), 0, NULL, events );

    if (err != CL_SUCCESS) {

		releaseMemObjects(bufA, bufB, bufC);
        deleteBuffers<T>(A, B, C, backC);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::HEMM() failed";
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

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufC, CL_TRUE, 0,
        (lengthC + params->offCY) * sizeof(T), backC, 0,
        NULL, NULL);
    if (err != CL_SUCCESS)
    {
        ::std::cerr << "WARNING: corr-hemm: Erorr reading buffer..." << err << ::std::endl;
    }
    //printMatrixBlock( params->order, 0, 0, params->M, params->N, params->ldc, backC);

    releaseMemObjects(bufA, bufB, bufC);

    // handle lda correctly based on row-major/col-major..
    compareMatrices<T>(params->order, params->M , params->N, (C + params->offCY), (backC + params->offCY), params->ldc);
    deleteBuffers<T>(A, B, C, backC);
    delete[] events;
}

// Instantiate the test

TEST_P(HEMM, chemm) {
    TestParams params;

    getParams(&params);
    hemmCorrectnessTest<FloatComplex>(&params);
}

TEST_P(HEMM, zhemm) {
    TestParams params;

    getParams(&params);
    hemmCorrectnessTest<DoubleComplex>(&params);
}
