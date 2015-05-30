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
#include <her.h>

static void
releaseMemObjects(cl_mem objA, cl_mem objX)
{

    if( objA!=NULL)
    clReleaseMemObject(objA);
    if( objX!=NULL)
    clReleaseMemObject(objX);
}

template <typename T> static void
deleteBuffers(T *A, T *X, T *backA)
{
	if(A != NULL)
	{
    delete[] A;
	}
	if(X != NULL)
	{
    delete[] X;
	}
	if(backA != NULL)
	{
    delete[] backA;
}
}

template <typename T>
void
herCorrectnessTest(TestParams *params)
{
    cl_int err;
    T *A, *X, *backA;
	T alpha_;
    cl_mem bufA, bufX;
    clMath::BlasBase *base;
    cl_event *events;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(DoubleComplex)) &&
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

    size_t lengthA = params->N * params->lda;
    size_t lengthX = (1 + ((params->N -1) * abs(params->incx)));
    alpha_ = convertMultiplier<T>(params->alpha);

    A 	    = new T[lengthA + params->offa ];
    backA 	= new T[lengthA + params->offa ];
    X		= new T[lengthX + params->offBX ];

	if((A == NULL) || (backA == NULL) || (X == NULL))
    {
        ::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        deleteBuffers<T>(A, X, backA);
        delete[] events;
		SUCCEED();
        return;
    }
    srand(params->seed);

    ::std::cerr << "Generating input data... ";
	randomHerMatrices( params->order, params->uplo, params->N, &alpha_, (A + params->offa), params->lda, (X + params->offBX), params->incx );
    memcpy(backA, A, (lengthA + params->offa)* sizeof(*A));
	::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufA = base->createEnqueueBuffer(A, (lengthA + params->offa) * sizeof(*A), 0, CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(X, (lengthX + params->offBX) * sizeof(*X), 0, CL_MEM_READ_ONLY);

    ::std::cerr << "Calling reference xHER routine... ";

    clblasOrder fOrder;
    clblasUplo fUplo;
    fOrder = params->order;
    fUplo = params->uplo;

    if (fOrder != clblasColumnMajor) {

        doConjugate( (X + params->offBX), (1 + (params->N-1) * abs(params->incx)), 1, 1 );
        fOrder = clblasColumnMajor;
		fUplo = (fUplo == clblasLower)? clblasUpper : clblasLower;
	}
	clMath::blas::her( fOrder, fUplo, params->N, CREAL(alpha_), X , params->offBX, params->incx, A, params->offa, params->lda );
    ::std::cerr << "Done" << ::std::endl;

    if ((bufA == NULL) || (bufX == NULL) ) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(backA, A, X);
        delete[] events;
		if(bufA == NULL)
		{
			::std::cerr << "BufA is null, lengthA is " << lengthA << ::std::endl;
		}
		if(bufX == NULL)
		{
			::std::cerr << "BufX is null, lengthX is  " << lengthX << ::std::endl;
		}

        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xHER routine... ";

    err = (cl_int)::clMath::clblas::her( params->order, params->uplo, params->N, CREAL(alpha_),
						bufX, params->offBX, params->incx, bufA, params->offa, params->lda,
						params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(backA, A, X);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::HER() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufA, bufX);
        deleteBuffers<T>(backA, A, X);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;

    err = clEnqueueReadBuffer(base->commandQueues()[0], bufA, CL_TRUE, 0,
        (lengthA + params->offa) * sizeof(*A), backA, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "HER: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufA, bufX);

	printf("Comparing the results\n");
	compareMatrices<T>(params->order, params->N , params->N, (A + params->offa), (backA + params->offa),
                       params->lda);

	deleteBuffers<T>( A, backA, X);
    delete[] events;
}

// Instantiate the test

TEST_P(HER, cher) {
    TestParams params;

    getParams(&params);
    herCorrectnessTest<FloatComplex>(&params);
}

TEST_P(HER, zher) {
    TestParams params;

    getParams(&params);
    herCorrectnessTest<DoubleComplex>(&params);
}
