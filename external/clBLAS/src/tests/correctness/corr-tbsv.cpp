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
#include <tbsv.h>
#include "trsv-delta.h"

static void
releaseMemObjects(cl_mem objA, cl_mem objX)
{
    if(objA != NULL)
	{
        clReleaseMemObject(objA);
    }
	if(objX != NULL)
	{
        clReleaseMemObject(objX);
	}

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
tbsvCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *blasX, *clblasX;
    cl_mem bufA, bufX;
  	cl_double *deltaX;
    clMath::BlasBase *base;
    cl_event *events;
	size_t lengthX, lengthA;

    base = clMath::BlasBase::getInstance();

    if (( (typeid(T) == typeid(DoubleComplex)) || (typeid(T) == typeid(cl_double)) ) &&
        !base->isDevSupportDoublePrecision()) {

        std::cerr << ">> WARNING: The target device doesn't support native "
                     "double precision floating point arithmetic" <<
                     std::endl << ">> Test skipped" << std::endl;
        SUCCEED();
        return;
    }

	printf("number of command queues : %d\n\n", params->numCommandQueues);

    events = new cl_event[params->numCommandQueues];
    memset(events, 0, params->numCommandQueues * sizeof(cl_event));

    lengthA = params->N  * params->lda ;
    lengthX = (params->N - 1)*abs(params->incx) + 1;

    A 	= new T[ lengthA + params->offA ];
    blasX  		= new T[ lengthX + params->offBX ];
	clblasX 	= new T[ lengthX + params->offBX ];
	deltaX	= new cl_double[lengthX + params->offBX];

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

	if((A == NULL) || (blasX == NULL) || (clblasX == NULL))
	{
		deleteBuffers<T>(A, blasX, clblasX, deltaX);
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped!!!!!!!!!!!!" << ::std::endl;
        delete[] events;
        SUCCEED();
        return;
	}
	memset( deltaX, 0, (lengthX + params->offBX)*sizeof(cl_double) );
    memset( blasX, 0, (lengthX + params->offBX)*sizeof(T));
    memset( clblasX, 0, (lengthX + params->offBX)*sizeof(T));

	randomTbsvMatrices<T>( params->order, params->uplo, params->diag, params->N, params->K,
	                        (A + params->offA), params->lda, (blasX + params->offBX), params->incx );

	// Generate delta X for result comparison
	tbsvDelta<T>( params->order, params->uplo, params->transA, params->diag, params->N, params->K,
	                (A + params->offA), params->lda, (blasX + params->offBX), params->incx, (deltaX + params->offBX) );

    memcpy(clblasX, blasX, (lengthX + params->offBX)* sizeof(*blasX));
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offA)* sizeof(*A), 0, CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(blasX, (lengthX + params->offBX)* sizeof(*blasX), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xTBSV routine... ";

	clblasOrder fOrder;
	clblasTranspose fTrans;
    clblasUplo fUplo;
	fOrder = params->order;
	fTrans = params->transA;
    fUplo = params->uplo;
	size_t  fN = params->N, fK = params->K;

	if (fOrder != clblasColumnMajor)
    {
        fOrder = clblasColumnMajor;
        fTrans = (params->transA == clblasNoTrans)? clblasTrans : clblasNoTrans;
        fUplo = (params->uplo == clblasLower)? clblasUpper : clblasLower;

        if( params->transA == clblasConjTrans )
            doConjugate( (A + params->offA), params->N, params->lda, params->lda );
   	}

	clMath::blas::tbsv(fOrder, fUplo, fTrans, params->diag, fN, fK, A, params->offA, params->lda, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufX == NULL)) {
        // Skip the test, the most probable reason is
        //     matrix too big for a device.

        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(A, blasX, clblasX, deltaX);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTBSV routine... ";
    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT:( typeid(T) == typeid(cl_double))? TYPE_DOUBLE:
                                      ( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    err = (cl_int)clMath::clblas::tbsv(type, params->order, params->uplo, params->transA, params->diag, params->N, params->K,
                                            bufA, params->offA, params->lda, bufX, params->offBX, params->incx,
                                            params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(A, blasX, clblasX, deltaX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TBSV() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(A, blasX, clblasX, deltaX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
                                (lengthX + params->offBX) * sizeof(*clblasX), clblasX, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "TBSV: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufA, bufX);
    compareMatrices<T>(clblasColumnMajor, lengthX , 1, (blasX + params->offBX), (clblasX + params->offBX),
                       lengthX, (deltaX + params->offBX) );
    deleteBuffers<T>(A, blasX, clblasX, deltaX);
    delete[] events;
}

// Instantiate the test

TEST_P(TBSV, stbsv) {
    TestParams params;

    getParams(&params);
    tbsvCorrectnessTest<cl_float>(&params);
}

TEST_P(TBSV, dtbsv) {
    TestParams params;

    getParams(&params);
    tbsvCorrectnessTest<cl_double>(&params);
}

TEST_P(TBSV, ctbsv) {
    TestParams params;

    getParams(&params);
    tbsvCorrectnessTest<FloatComplex>(&params);
}

TEST_P(TBSV, ztbsv) {
    TestParams params;

    getParams(&params);
    tbsvCorrectnessTest<DoubleComplex>(&params);
}
