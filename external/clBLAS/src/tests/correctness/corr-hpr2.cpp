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
#include <hpr2.h>

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
deleteBuffers(T *blasA, T *clblasA, T *X, T *Y)
{
	if(blasA != NULL)
    {
    delete[] blasA;
    }
	if(clblasA != NULL)
	{
    delete[] clblasA;
	}
	if(X != NULL)
	{
	delete[] X;
	}
	if(Y != NULL)
	{
	delete[] Y;
}
}

template <typename T>
void
hpr2CorrectnessTest(TestParams *params)
{
    cl_int err;
    T *blasAP, *clblasAP, *X, *Y;
    cl_mem bufAP, bufX, bufY;
    clMath::BlasBase *base;
    cl_event *events;
	T alpha;

    base = clMath::BlasBase::getInstance();

    if ((typeid(T) == typeid(cl_double2)) &&
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

    size_t lengthAP = (params->N *( params->N + 1 ))/2 ;
    size_t lengthX = (1 + ((params->N - 1) * abs(params->incx)));
	size_t lengthY = (1 + ((params->N - 1) * abs(params->incy)));

    blasAP 		= new T[lengthAP + params->offa ];
    clblasAP 	= new T[lengthAP + params->offa ];
    X		 	= new T[lengthX + params->offBX ];
	Y			= new T[lengthY + params->offCY ];

    srand(params->seed);

	if((blasAP == NULL) || (clblasAP == NULL) || (X == NULL) || (Y == NULL))
	{
		deleteBuffers<T>(blasAP, clblasAP, X, Y);
		::std::cerr << "Cannot allocate memory on host side\n" << "!!!!!!!!!!!!Test skipped.!!!!!!!!!!!!" << ::std::endl;
        delete[] events;
        SUCCEED();
        return;
	}
	alpha =  convertMultiplier<T>(params->alpha);

    ::std::cerr << "Generating input data... ";
    randomHer2Matrices<T>(params->order, params->uplo, params->N, &alpha, (blasAP + params->offa), params->lda,
							(X + params->offBX), params->incx, (Y + params->offCY), params->incy);

	// Copy blasA to clblasA
    memcpy(clblasAP, blasAP, (lengthAP + params->offa)* sizeof(*blasAP));
    ::std::cerr << "Done" << ::std::endl;

	// Allocate buffers
    bufAP = base->createEnqueueBuffer(clblasAP, (lengthAP + params->offa)* sizeof(*clblasAP), 0,CL_MEM_READ_WRITE);
    bufX = base->createEnqueueBuffer(X, (lengthX + params->offBX)* sizeof(*X), 0, CL_MEM_READ_ONLY);
	bufY = base->createEnqueueBuffer(Y, (lengthY + params->offCY)* sizeof(*Y), 0, CL_MEM_READ_ONLY);

    ::std::cerr << "Calling reference xHPR2 routine... ";

	clblasOrder order;
    clblasUplo fUplo;
	order = params->order;
    fUplo = params->uplo;

	if (order != clblasColumnMajor)
    {
		doConjugate( (X + params->offBX), 1, (1 + (params->N-1) * abs(params->incx)), (1 + (params->N-1) * abs(params->incx)) );
        doConjugate( (Y + params->offCY), 1, (1 + (params->N-1) * abs(params->incy)), (1 + (params->N-1) * abs(params->incy)) );
        order = clblasColumnMajor;
        fUplo = (fUplo == clblasLower)? clblasUpper : clblasLower;
		::clMath::blas::hpr2( order, fUplo, params->N, alpha, Y, params->offCY, params->incy, X, params->offBX, params->incx, blasAP, params->offa);
    }
	else {
		::clMath::blas::hpr2( order, fUplo, params->N, alpha, X, params->offBX, params->incx, Y, params->offCY, params->incy, blasAP, params->offa);
	}
    ::std::cerr << "Done" << ::std::endl;

    if ((bufAP == NULL) || (bufX == NULL) || (bufY == NULL)) {
        /* Skip the test, the most probable reason is
         *     matrix too big for a device.
         */
        releaseMemObjects(bufAP, bufX, bufY);
        deleteBuffers<T>(blasAP, clblasAP, X, Y);
        delete[] events;
        ::std::cerr << ">> Failed to create/enqueue buffer for a matrix."
            << ::std::endl
            << ">> Can't execute the test, because data is not transfered to GPU."
            << ::std::endl
            << ">> Test skipped." << ::std::endl;
        SUCCEED();
        return;
    }

    ::std::cerr << "Calling clblas xHPR2 routine... ";

    err = (cl_int)::clMath::clblas::hpr2( params->order, params->uplo, params->N, alpha,
						bufX, params->offBX, params->incx, bufY, params->offCY, params->incy, bufAP, params->offa,
						params->numCommandQueues, base->commandQueues(),
    					0, NULL, events);

    if (err != CL_SUCCESS) {
        releaseMemObjects(bufAP, bufX, bufY);
        deleteBuffers<T>(blasAP, clblasAP, X, Y);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "::clMath::clblas::HPR2() failed";
    }

    err = waitForSuccessfulFinish(params->numCommandQueues,
        base->commandQueues(), events);
    if (err != CL_SUCCESS) {
        releaseMemObjects(bufAP, bufX, bufY);
        deleteBuffers<T>(blasAP, clblasAP, X, Y);
        delete[] events;
        ASSERT_EQ(CL_SUCCESS, err) << "waitForSuccessfulFinish()";
    }
    ::std::cerr << "Done" << ::std::endl;


    err = clEnqueueReadBuffer(base->commandQueues()[0], bufAP, CL_TRUE, 0,
        (lengthAP + params->offa) * sizeof(*clblasAP), clblasAP, 0,
        NULL, NULL);
	if (err != CL_SUCCESS)
	{
		::std::cerr << "HPR2: Reading results failed...." << std::endl;
	}

    releaseMemObjects(bufAP, bufX, bufY);

    compareMatrices<T>(clblasColumnMajor, lengthAP, 1, (blasAP + params->offa), (clblasAP + params->offa), lengthAP);

	deleteBuffers<T>(blasAP, clblasAP, X, Y);
    delete[] events;
}

// Instantiate the test

TEST_P(HPR2, chpr2) {
    TestParams params;

    getParams(&params);
    hpr2CorrectnessTest<FloatComplex>(&params);
}

TEST_P(HPR2, zhpr2) {
    TestParams params;

    getParams(&params);
    hpr2CorrectnessTest<DoubleComplex>(&params);
}
