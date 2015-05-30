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
#include <tpsv.h>
#include <cltypes.h>

#include "trsv-delta.h"

static void
releaseMemObjects(cl_mem objA, cl_mem objX)
{
	if (objA != NULL)
    clReleaseMemObject(objA);
	if (objX != NULL)
    clReleaseMemObject(objX);
}

template <typename T> static void
deleteBuffers(T *A, T *blasX, T *backX, cl_double *deltaX)
{
    if( A != NULL )
	{
    delete[] A;
	}
	if( blasX != NULL )
	{
    delete[] blasX;
	}
	if( backX != NULL )
	{
		delete[] backX;
	}
	if( deltaX != NULL )
	{
	delete[] deltaX;
}
}

template <typename T>
void
tpsvCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *blasX, *backX;
	cl_double *deltaX;
    cl_mem bufA, bufX;
    clMath::BlasBase *base;
    cl_event *events;

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

    size_t lengthA = (params->N * (params->N + 1)) / 2;
    size_t lengthX = (1 + ((params->N -1) * abs(params->incx)));

    A 		= new T[lengthA + params->offa];
    blasX 	= new T[lengthX + params->offBX];
    backX 	= new T[lengthX + params->offBX];
	deltaX	= new cl_double[lengthX + params->offBX];

	if ((A==NULL) || (blasX == NULL) || (backX == NULL) || (deltaX == NULL))
	{
		::std::cerr << "Unable to allocate matrices in Host memory" << std::endl;
		deleteBuffers<T>(A, blasX, backX, deltaX);
		delete[] events;
		SUCCEED();
		return;
	}
	memset( deltaX, 0, lengthX*sizeof(cl_double) );
	memset( blasX, 0, lengthX*sizeof(T) );

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

	//custom generation function in blas-random.h
	randomTrsvMatrices<T>( params->order, params->uplo, params->diag, params->N, (A + params->offa), 0, (blasX + params->offBX), params->incx);

	// Generate delta X for result comparison
	trsvDelta<T>( params->order, params->uplo, params->transA, params->diag, params->N, (A + params->offa), 0, (blasX + params->offBX), params->incx, (deltaX + params->offBX) );

	/*printf("\n\n before acml call\nA\n");
	printMatrixBlock( params->order, 0, 0, params->N, params->N, params->lda, A);
	printf("\nX\n");
	printMatrixBlock( clblasColumnMajor, 0, 0, lengthX, 1, lengthX, blasX);*/

    // Copy blasX to clblasX
    memcpy(backX, blasX, (lengthX + params->offBX) * sizeof(T));
	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offa)* sizeof(T), 0, CL_MEM_READ_ONLY);
    bufX = base->createEnqueueBuffer(backX, (lengthX + params->offBX)* sizeof(T), 0, CL_MEM_WRITE_ONLY);
    ::std::cerr << "Done" << ::std::endl;

    ::std::cerr << "Calling reference xTPSV routine... ";

    clblasOrder order;
    clblasUplo fUplo;
    clblasTranspose fTrans;

    order = params->order;
    fUplo = params->uplo;
    fTrans = params->transA;


    if (order != clblasColumnMajor)
    {
        order = clblasColumnMajor;
        fUplo =  (params->uplo == clblasUpper)? clblasLower : clblasUpper;
        fTrans = (params->transA == clblasNoTrans)? clblasTrans : clblasNoTrans;

        if( params->transA == clblasConjTrans )
            doConjugate((A + params->offa), 1, lengthA, 1);
    }
	::clMath::blas::tpsv( order, fUplo, fTrans, params->diag, params->N, A, params->offa, blasX, params->offBX, params->incx);
	::std::cerr << "Done" << ::std::endl;

	/*
	printf("\n\n acml result X\n");
	printf("\nblasX\n");
	printMatrixBlock( clblasColumnMajor, 0, 0, lengthX, 1, lengthX, blasX);*/

    if ((bufA == NULL) || (bufX == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(A, blasX, backX, deltaX);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTPSV routine... ";

    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT : ( typeid(T) == typeid(cl_double))? TYPE_DOUBLE: ( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    // Should use bufXTemp as well
    err = (cl_int)::clMath::clblas::tpsv(type, params->order, params->uplo, params->transA, params->diag, params->N, bufA,
    					params->offa, bufX, params->offBX, params->incx, params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {

        deleteBuffers<T>(A, blasX, backX, deltaX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TPSV() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {

        deleteBuffers<T>(A, blasX, backX, deltaX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
        lengthX * sizeof(*backX), backX, 0,
        NULL, NULL);

    releaseMemObjects(bufA, bufX);

	/*
	printf("\n\n clblas result X\n");
	printf("\nclBlasX\n");
	printMatrixBlock( clblasColumnMajor, 0, 0, lengthX, 1, lengthX, backX);

	printf("\n\n delta X\n\n");
	printMatrixBlock( clblasColumnMajor, 0, 0, lengthX, 1, lengthX, deltaX);*/

    // handle lda correctly based on row-major/col-major..
    compareMatrices<T>( clblasColumnMajor, lengthX , 1, blasX, backX,
                       lengthX, deltaX );
    deleteBuffers<T>(A, blasX, backX, deltaX);
    delete[] events;
}

// Instantiate the test

TEST_P(TPSV, stpsv) {
    TestParams params;

    getParams(&params);
    tpsvCorrectnessTest<cl_float>(&params);
}

TEST_P(TPSV, dtpsv) {
    TestParams params;

    getParams(&params);
    tpsvCorrectnessTest<cl_double>(&params);
}

TEST_P(TPSV, ctpsv) {
    TestParams params;

    getParams(&params);
    tpsvCorrectnessTest<FloatComplex>(&params);
}

TEST_P(TPSV, ztpsv) {
    TestParams params;

    getParams(&params);
    tpsvCorrectnessTest<DoubleComplex>(&params);
}
