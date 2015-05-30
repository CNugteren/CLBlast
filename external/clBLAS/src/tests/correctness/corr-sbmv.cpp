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
#include <sbmv.h>
#include <gbmv.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objX, cl_mem objY)
{
    if(objA != NULL)
	{
        clReleaseMemObject(objA);
    }
	if(objX != NULL)
	{
        clReleaseMemObject(objX);
	}
	if(objY != NULL)
	{
	    clReleaseMemObject(objY);
    }
}

template <typename T> static void
deleteBuffers(T *A, T *X, T *blasY, T *clblasY)
{
    if(A != NULL)
	{
        delete[] A;
	}
    if(X != NULL)
	{
        delete[] X;
	}
	if(blasY != NULL)
	{
	    delete[] blasY;
	}
    if(clblasY != NULL)
	{
        delete[] clblasY; // To hold clblas GBMV call results
    }
}

template <typename T>
void
sbmvCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *X, *blasY, *clblasY;
    cl_mem bufA, bufX, bufY;
    clMath::BlasBase *base;
    cl_event *events;
	T alpha, beta;
	size_t lengthX, lengthY, lengthA;

    base = clMath::BlasBase::getInstance();

    if (((typeid(T) == typeid(cl_double))) &&
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

    lengthA =  params->N  * params->lda;
    lengthX = (params->N - 1)*abs(params->incx) + 1;
    lengthY = (params->N - 1)*abs(params->incy) + 1;

    A 	= new T[ lengthA + params->offA ];
    X 	= new T[ lengthX + params->offBX ];
    blasY  		= new T[ lengthY + params->offCY ];
	clblasY 	= new T[ lengthY + params->offCY ];

    srand(params->seed);

    ::std::cerr << "Generating input data... ";

	if((A == NULL) || (X == NULL) || (blasY == NULL) || (clblasY == NULL))
	{
		deleteBuffers<T>(A, X, blasY, clblasY);
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        delete[] events;
        SUCCEED();
        return;
	}

	alpha = convertMultiplier<T>(params->alpha);
	beta = convertMultiplier<T>(params->beta);

    randomGbmvMatrices(params->order, clblasNoTrans, params->N, params->N, &alpha, &beta,
                        (A + params->offA), params->lda, (X+params->offBX), params->incx, (blasY+params->offCY), params->incy );
    // Copy blasY to clblasY
    memcpy(clblasY, blasY, (lengthY + params->offCY)* sizeof(*blasY));
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offA)* sizeof(*A), 0, CL_MEM_READ_ONLY);
    bufX = base->createEnqueueBuffer(X, (lengthX + params->offBX)* sizeof(*X), 0, CL_MEM_READ_ONLY);
    bufY = base->createEnqueueBuffer(clblasY, (lengthY + params->offCY) * sizeof(*clblasY), 0, CL_MEM_READ_WRITE);

    ::std::cerr << "Calling reference xSBMV routine... ";

	clblasOrder fOrder;
	clblasUplo fUplo;
	fOrder = params->order;
	fUplo = params->uplo;
	size_t fN = params->N, fK = params->K;

	if (fOrder != clblasColumnMajor)
    {
        fOrder = clblasColumnMajor;
        fUplo = (params->uplo == clblasLower)? clblasUpper : clblasLower;
        fN = params->N;
   	}

	clMath::blas::sbmv(fOrder, fUplo, fN, fK, alpha, A, params->offA, params->lda,
							X, params->offBX, params->incx, beta, blasY, params->offCY, params->incy);
    ::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufX == NULL) || (bufY == NULL)) {
        // Skip the test, the most probable reason is
        //     matrix too big for a device.

        releaseMemObjects(bufA, bufX, bufY);
        deleteBuffers<T>(A, X, blasY, clblasY);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xSBMV routine... ";

    err = (cl_int)clMath::clblas::sbmv(params->order, params->uplo, params->N, params->K,
                                        alpha, bufA, params->offA, params->lda, bufX, params->offBX, params->incx,
                                        beta, bufY, params->offCY, params->incy,
                                        params->numCommandQueues, base->commandQueues(), 0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX, bufY);
        deleteBuffers<T>(A, X, blasY, clblasY);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::SBMV() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX, bufY);
        deleteBuffers<T>(A, X, blasY, clblasY);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufY, CL_TRUE, 0,
        (lengthY + params->offCY) * sizeof(*clblasY), clblasY, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "SBMV: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufA, bufX, bufY);
    compareMatrices<T>(clblasColumnMajor, lengthY , 1, (blasY + params->offCY), (clblasY + params->offCY),
                       lengthY);
    deleteBuffers<T>(A, X, blasY, clblasY);
    delete[] events;
}

// Instantiate the test

TEST_P(SBMV, ssbmv) {
    TestParams params;

    getParams(&params);
    sbmvCorrectnessTest<cl_float>(&params);
}

TEST_P(SBMV, dsbmv) {
    TestParams params;

    getParams(&params);
    sbmvCorrectnessTest<cl_double>(&params);
}

