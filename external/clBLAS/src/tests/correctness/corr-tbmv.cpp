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
#include <tbmv.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objX, cl_mem objXtemp)
{
    if(objA != NULL)
	{
        clReleaseMemObject(objA);
    }
	if(objX != NULL)
	{
        clReleaseMemObject(objX);
	}
    if(objXtemp != NULL)
    {
        clReleaseMemObject(objXtemp);
    }

}

template <typename T> static void
deleteBuffers(T *A, T *blasX, T *clblasX)
{
    if(A != NULL)
	{
        delete[] A;
	}
    if(blasX != NULL)
	{
        delete[] blasX;
	}
    if(clblasX != NULL)
	{
        delete[] clblasX; // To hold clblas TBMV call results
    }
}

template <typename T>
void
tbmvCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *blasX, *clblasX;
    cl_mem bufA, bufX, bufXtemp;
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

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

	if((A == NULL) || (blasX == NULL) || (clblasX == NULL))
	{
		deleteBuffers<T>(A, blasX, clblasX);
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped!!!!!!!!!!!!" << ::std::endl;
        delete[] events;
        SUCCEED();
        return;
	}
	randomTbmvMatrices( params->N, (A + params->offA), params->lda, (blasX + params->offBX), params->incx );

    // Copy blasY to clblasY
    memcpy(clblasX, blasX, (lengthX + params->offBX)* sizeof(*blasX));
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offA)* sizeof(*A), 0, CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(blasX, (lengthX + params->offBX)* sizeof(*blasX), 0, CL_MEM_READ_WRITE);
    bufXtemp = base->createEnqueueBuffer(blasX, (lengthX + params->offBX)* sizeof(*blasX), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xTBMV routine... ";

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
            doConjugate( (A + params->offA), 1, lengthA, params->lda );
   	}

	clMath::blas::tbmv(fOrder, fUplo, fTrans, params->diag, fN, fK, A, params->offA, params->lda, blasX, params->offBX, params->incx);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufX == NULL)|| (bufXtemp == NULL)) {
        // Skip the test, the most probable reason is
        //     matrix too big for a device.

        releaseMemObjects(bufA, bufX, bufXtemp );
        deleteBuffers<T>(A, blasX, clblasX);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xTBMV routine... ";
    DataType type;
    type = ( typeid(T) == typeid(cl_float))? TYPE_FLOAT:( typeid(T) == typeid(cl_double))? TYPE_DOUBLE:
                                      ( typeid(T) == typeid(cl_float2))? TYPE_COMPLEX_FLOAT: TYPE_COMPLEX_DOUBLE;

    err = (cl_int)clMath::clblas::tbmv(type, params->order, params->uplo, params->transA, params->diag, params->N, params->K,
                                        bufA, params->offA, params->lda, bufX, params->offBX, params->incx, bufXtemp,
                                        params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX, bufXtemp);
        deleteBuffers<T>(A, blasX, clblasX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::TBMV() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX, bufXtemp);
        deleteBuffers<T>(A, blasX, clblasX);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufX, CL_TRUE, 0,
        (lengthX + params->offBX) * sizeof(*clblasX), clblasX, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "TBMV: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufA, bufX, bufXtemp);
    compareMatrices<T>(clblasColumnMajor, lengthX , 1, (blasX + params->offBX), (clblasX + params->offBX),
                       lengthX);
    deleteBuffers<T>(A, blasX, clblasX);
    delete[] events;
}

// Instantiate the test

TEST_P(TBMV, stbmv) {
    TestParams params;

    getParams(&params);
    tbmvCorrectnessTest<cl_float>(&params);
}

TEST_P(TBMV, dtbmv) {
    TestParams params;

    getParams(&params);
    tbmvCorrectnessTest<cl_double>(&params);
}

TEST_P(TBMV, ctbmv) {
    TestParams params;

    getParams(&params);
    tbmvCorrectnessTest<FloatComplex>(&params);
}

TEST_P(TBMV, ztbmv) {
    TestParams params;

    getParams(&params);
    tbmvCorrectnessTest<DoubleComplex>(&params);
}
